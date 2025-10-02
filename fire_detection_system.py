#!/usr/bin/env python3
"""
Fire and Smoke Detection System using YOLOv8
Model: https://github.com/luminous0219/fire-and-smoke-detection-yolov8
YOLOv8n model trained with 150 epochs for fire and smoke detection
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from ultralytics import YOLO
import os


class FireDetectionSystem:
    """Fire and smoke detection using YOLOv8n trained model"""

    def __init__(self, model_path: str = "models/fire_smoke_best.pt"):
        """
        Initialize Fire Detection System
        
        Args:
            model_path: Path to the YOLOv8 fire detection model
        """
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Check if model exists
        if not Path(model_path).exists():
            print(f"⚠️  Model not found at: {model_path}")
            print("📥 Please download the model from:")
            print("   https://github.com/luminous0219/fire-and-smoke-detection-yolov8")
            print(f"   And place 'best.pt' at: {model_path}")
            raise FileNotFoundError(f"Fire detection model not found at {model_path}")
        
        print(f"🔥 Loading fire detection model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Class names for this specific model (fire and smoke)
        # The model from the GitHub repo detects: fire (0) and smoke (1)
        self.class_names = {
            0: 'fire',
            1: 'smoke'
        }
        
        # Detection thresholds
        self.confidence_threshold = 0.3  # Lower threshold for better detection
        self.iou_threshold = 0.45  # IoU threshold for NMS
        
        # Alert cooldown (seconds) - to avoid spamming alerts
        self.alert_cooldown = 3.0
        self.last_alert_time = {}
        
        # Statistics tracking
        self.stats = {
            'total_detections': 0,
            'fire_detections': 0,
            'smoke_detections': 0,
            'critical_alerts': 0
        }
        
        print("✅ Fire detection system initialized successfully!")
        
    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect fire and smoke in a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of detection dictionaries
        """
        h, w = frame.shape[:2]
        detections = []
        
        # Run detection with the model
        results = self.model.predict(
            frame, 
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False  # Suppress output
        )[0]
        
        current_time = datetime.now()
        
        # Process detections
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Ensure coordinates are within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Get class name
                class_name = self.class_names.get(cls, f'unknown_{cls}')
                
                # Calculate detection area
                detection_area = (x2 - x1) * (y2 - y1)
                frame_area = w * h
                
                # Calculate severity
                severity = self._calculate_severity(
                    class_name, conf, detection_area, frame_area
                )
                
                # Check if we should alert (cooldown logic)
                should_alert = False
                alert_key = f"{class_name}_{x1}_{y1}_{x2}_{y2}"
                
                if alert_key not in self.last_alert_time:
                    should_alert = True
                    self.last_alert_time[alert_key] = current_time
                else:
                    time_diff = (current_time - self.last_alert_time[alert_key]).total_seconds()
                    if time_diff > self.alert_cooldown:
                        should_alert = True
                        self.last_alert_time[alert_key] = current_time
                
                # Update statistics
                self.stats['total_detections'] += 1
                if class_name == 'fire':
                    self.stats['fire_detections'] += 1
                elif class_name == 'smoke':
                    self.stats['smoke_detections'] += 1
                
                if severity == 'critical':
                    self.stats['critical_alerts'] += 1
                
                # Create detection dictionary
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'class': class_name,
                    'confidence': conf,
                    'timestamp': current_time,
                    'alert': should_alert,
                    'severity': severity,
                    'area_ratio': detection_area / frame_area
                }
                
                detections.append(detection)
        
        return detections
    
    def _calculate_severity(self, class_name: str, confidence: float, 
                           detection_area: int, frame_area: int) -> str:
        """
        Calculate severity level based on detection parameters
        
        Args:
            class_name: 'fire' or 'smoke'
            confidence: Detection confidence
            detection_area: Area of detection in pixels
            frame_area: Total frame area in pixels
            
        Returns:
            Severity level: 'low', 'medium', 'high', 'critical'
        """
        area_ratio = detection_area / frame_area
        
        # Fire is more critical than smoke
        if class_name == 'fire':
            if confidence > 0.7 and area_ratio > 0.15:
                return 'critical'
            elif confidence > 0.6 and area_ratio > 0.08:
                return 'high'
            elif confidence > 0.5 or area_ratio > 0.05:
                return 'medium'
            else:
                return 'low'
        else:  # smoke
            if confidence > 0.7 and area_ratio > 0.25:
                return 'high'
            elif confidence > 0.6 and area_ratio > 0.12:
                return 'medium'
            else:
                return 'low'
    
    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            
        Returns:
            Frame with drawn detections
        """
        output = frame.copy()
        
        # Define colors for different severity levels
        severity_colors = {
            'critical': (0, 0, 255),      # Red
            'high': (0, 100, 255),        # Orange-Red
            'medium': (0, 165, 255),      # Orange
            'low': (0, 255, 255)          # Yellow
        }
        
        # Define colors for different classes
        class_colors = {
            'fire': (0, 0, 255),          # Red
            'smoke': (128, 128, 128)      # Gray
        }
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            severity = detection['severity']
            area_ratio = detection.get('area_ratio', 0)
            
            # Choose color based on severity
            color = severity_colors.get(severity, (0, 255, 0))
            
            # Make critical alerts more visible
            thickness = 4 if severity == 'critical' else 2
            
            # Draw rectangle
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare labels
            label = f"{class_name.upper()}"
            conf_label = f"Conf: {confidence:.2f}"
            severity_label = f"Severity: {severity.upper()}"
            area_label = f"Area: {area_ratio*100:.1f}%"
            
            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            # Draw background rectangle for text
            text_y = y1 - 10
            if text_y < 80:  # If too close to top, draw below
                text_y = y2 + 20
                label_positions = [
                    (text_y, label),
                    (text_y + 20, conf_label),
                    (text_y + 40, severity_label),
                    (text_y + 60, area_label)
                ]
            else:
                label_positions = [
                    (text_y, label),
                    (text_y - 20, conf_label),
                    (text_y - 40, severity_label),
                    (text_y - 60, area_label)
                ]
            
            # Draw background for all text
            max_text_width = max([cv2.getTextSize(text, font, font_scale, font_thickness)[0][0] 
                                 for _, text in label_positions])
            
            bg_y1 = min([y for y, _ in label_positions]) - 5
            bg_y2 = max([y for y, _ in label_positions]) + 5
            cv2.rectangle(output, (x1, bg_y1), (x1 + max_text_width + 10, bg_y2), 
                         color, -1)
            
            # Draw all text labels
            for y_pos, text in label_positions:
                cv2.putText(output, text, (x1 + 5, y_pos),
                           font, font_scale, (255, 255, 255), font_thickness)
            
            # Add alert icon if this is a new alert
            if detection.get('alert', False):
                alert_text = "!!! ALERT !!!"
                (alert_w, alert_h), _ = cv2.getTextSize(
                    alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3
                )
                alert_x = (x1 + x2 - alert_w) // 2
                alert_y = (y1 + y2) // 2
                
                # Draw alert background
                cv2.rectangle(output, 
                             (alert_x - 10, alert_y - alert_h - 10),
                             (alert_x + alert_w + 10, alert_y + 10),
                             (0, 0, 255), -1)
                
                # Draw alert text
                cv2.putText(output, alert_text, (alert_x, alert_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        # Draw overall warning if critical detections exist
        critical_detections = [d for d in detections if d['severity'] == 'critical']
        if critical_detections:
            warning_text = "CRITICAL FIRE ALERT!"
            (warn_w, warn_h), _ = cv2.getTextSize(
                warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
            )
            
            # Draw at top center
            warn_x = (output.shape[1] - warn_w) // 2
            cv2.rectangle(output, (warn_x - 20, 10), 
                         (warn_x + warn_w + 20, 50), (0, 0, 255), -1)
            cv2.putText(output, warning_text, (warn_x, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        return output
    
    def get_statistics(self) -> Dict:
        """
        Get detection statistics
        
        Returns:
            Dictionary with statistics
        """
        return {
            'model_loaded': self.model is not None,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'alert_cooldown': self.alert_cooldown,
            'active_alerts': len(self.last_alert_time),
            **self.stats
        }
    
    def reset_statistics(self):
        """Reset detection statistics"""
        self.stats = {
            'total_detections': 0,
            'fire_detections': 0,
            'smoke_detections': 0,
            'critical_alerts': 0
        }
        self.last_alert_time.clear()


# Quick demo
if __name__ == "__main__":
    import sys
    
    print("🔥 Fire Detection System - Demo")
    print("=" * 50)
    
    try:
        # Initialize system
        fire_system = FireDetectionSystem()
        
        # Open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            sys.exit(1)
        
        print("✅ Camera opened successfully")
        print("📹 Press 'q' to quit, 's' to show statistics")
        print("-" * 50)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect fire/smoke (process every frame for real-time detection)
            detections = fire_system.process_frame(frame)
            
            # Draw detections
            output = fire_system.draw_detections(frame, detections)
            
            # Add FPS counter
            cv2.putText(output, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display alerts in console
            if detections:
                for det in detections:
                    if det.get('alert', False):
                        severity_emoji = {
                            'critical': '🚨🚨🚨',
                            'high': '⚠️⚠️',
                            'medium': '⚠️',
                            'low': 'ℹ️'
                        }
                        emoji = severity_emoji.get(det['severity'], '⚠️')
                        print(f"{emoji} ALERT: {det['class'].upper()} detected! "
                              f"Confidence: {det['confidence']:.2f}, "
                              f"Severity: {det['severity'].upper()}, "
                              f"Area: {det['area_ratio']*100:.1f}%")
            
            # Show frame
            cv2.imshow("Fire & Smoke Detection", output)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Show statistics
                stats = fire_system.get_statistics()
                print("\n" + "=" * 50)
                print("📊 Detection Statistics:")
                print(f"  Total Detections: {stats['total_detections']}")
                print(f"  Fire Detections: {stats['fire_detections']}")
                print(f"  Smoke Detections: {stats['smoke_detections']}")
                print(f"  Critical Alerts: {stats['critical_alerts']}")
                print(f"  Active Alert Locations: {stats['active_alerts']}")
                print("=" * 50 + "\n")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        final_stats = fire_system.get_statistics()
        print("\n" + "=" * 50)
        print("🏁 Final Statistics:")
        print(f"  Total Frames Processed: {frame_count}")
        print(f"  Total Detections: {final_stats['total_detections']}")
        print(f"  Fire Detections: {final_stats['fire_detections']}")
        print(f"  Smoke Detections: {final_stats['smoke_detections']}")
        print(f"  Critical Alerts: {final_stats['critical_alerts']}")
        print("=" * 50)
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\n📥 Download Instructions:")
        print("1. Go to: https://github.com/luminous0219/fire-and-smoke-detection-yolov8")
        print("2. Download the 'best.pt' model file")
        print("3. Place it in the 'models' directory as 'fire_smoke_best.pt'")
        print("\nOr run:")
        print("  mkdir -p models")
        print("  # Download best.pt from GitHub and move it:")
        print("  mv best.pt models/fire_smoke_best.pt")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()