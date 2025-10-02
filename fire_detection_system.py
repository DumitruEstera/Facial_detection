#!/usr/bin/env python3
"""
Fire and Smoke Detection System using YOLOv8
Model: https://huggingface.co/kittendev/YOLOv8m-smoke-detection
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from ultralytics import YOLO
import os
import urllib.request


class FireDetectionSystem:
    """Fire and smoke detection using YOLOv8 trained model"""

    def __init__(self, model_path: str = "models/best.pt"):
        """
        Initialize Fire Detection System
        
        Args:
            model_path: Path to the YOLOv8 fire detection model
        """
        if model_path is None:
            model_path = self._download_model()
        
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Fire detection model '{model_path}' not found. "
                "Please download the model from HuggingFace or provide a valid path."
            )
        
        print(f"Loading fire detection model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Class names for fire detection (typically: fire, smoke)
        self.class_names = {
            0: 'fire',
            1: 'smoke'
        }
        
        # Detection threshold
        self.confidence_threshold = 0.5
        
        # Alert cooldown (seconds) - to avoid spamming alerts
        self.alert_cooldown = 5.0
        self.last_alert_time = {}
        
    def _download_model(self) -> str:
        """
        Download the fire detection model from HuggingFace if not present
        """
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "fire_detection.pt")
        
        if not os.path.exists(model_path):
            print("Downloading fire detection model from HuggingFace...")
            print("Note: You need to manually download the model from:")
            print("https://huggingface.co/kittendev/YOLOv8m-smoke-detection")
            print(f"And place it at: {model_path}")
            print("\nFor now, creating placeholder...")
            
            # For development, we'll try to download if URL is available
            # In production, user should manually download the model
            try:
                # This is a placeholder - user needs to download manually
                model_url = "https://huggingface.co/kittendev/YOLOv8m-smoke-detection/resolve/main/best.pt"
                print(f"Attempting to download from: {model_url}")
                urllib.request.urlretrieve(model_url, model_path)
                print("Model downloaded successfully!")
            except Exception as e:
                print(f"Could not auto-download model: {e}")
                print("Please download manually and place at:", model_path)
                raise FileNotFoundError(f"Model not found at {model_path}")
        
        return model_path
    
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
        
        # Run detection
        results = self.model.predict(frame, imgsz=640, conf=self.confidence_threshold)[0]
        
        current_time = datetime.now()
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Get class name
                class_name = self.class_names.get(cls, f'class_{cls}')
                
                # Check if we should alert (cooldown logic)
                should_alert = False
                alert_key = f"{class_name}_{x1}_{y1}"
                
                if alert_key not in self.last_alert_time:
                    should_alert = True
                    self.last_alert_time[alert_key] = current_time
                else:
                    time_diff = (current_time - self.last_alert_time[alert_key]).total_seconds()
                    if time_diff > self.alert_cooldown:
                        should_alert = True
                        self.last_alert_time[alert_key] = current_time
                
                # Create detection dictionary
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'class': class_name,
                    'confidence': conf,
                    'timestamp': current_time,
                    'alert': should_alert,
                    'severity': self._calculate_severity(class_name, conf, (x2-x1)*(y2-y1), w*h)
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
            if confidence > 0.8 and area_ratio > 0.2:
                return 'critical'
            elif confidence > 0.7 or area_ratio > 0.1:
                return 'high'
            elif confidence > 0.6:
                return 'medium'
            else:
                return 'low'
        else:  # smoke
            if confidence > 0.8 and area_ratio > 0.3:
                return 'high'
            elif confidence > 0.7 or area_ratio > 0.15:
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
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            severity = detection['severity']
            
            # Color based on class and severity
            if class_name == 'fire':
                if severity == 'critical':
                    color = (0, 0, 255)  # Red
                elif severity == 'high':
                    color = (0, 100, 255)  # Orange-Red
                else:
                    color = (0, 165, 255)  # Orange
            else:  # smoke
                if severity == 'high':
                    color = (128, 128, 128)  # Dark Gray
                elif severity == 'medium':
                    color = (169, 169, 169)  # Gray
                else:
                    color = (192, 192, 192)  # Light Gray
            
            # Draw rectangle
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            
            # Prepare label
            label = f"{class_name.upper()} ({confidence:.2f})"
            severity_label = f"Severity: {severity.upper()}"
            
            # Draw background for text
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            cv2.rectangle(output, (x1, y1 - text_height - 40), 
                         (x1 + max(text_width, 200), y1), color, -1)
            
            # Draw text
            cv2.putText(output, label, (x1 + 5, y1 - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(output, severity_label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add alert icon if this is a new alert
            if detection.get('alert', False):
                cv2.putText(output, "!!! ALERT !!!", (x1, y2 + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
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
            'alert_cooldown': self.alert_cooldown,
            'active_alerts': len(self.last_alert_time)
        }


# Quick demo
if __name__ == "__main__":
    import sys
    
    print("Fire Detection System - Demo")
    print("=" * 50)
    
    try:
        # Initialize system
        fire_system = FireDetectionSystem()
        
        # Open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            sys.exit(1)
        
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect fire/smoke
            detections = fire_system.process_frame(frame)
            
            # Draw detections
            output = fire_system.draw_detections(frame, detections)
            
            # Display alerts
            if detections:
                for det in detections:
                    if det.get('alert', False):
                        print(f"🚨 ALERT: {det['class'].upper()} detected! "
                              f"Confidence: {det['confidence']:.2f}, "
                              f"Severity: {det['severity'].upper()}")
            
            # Show frame
            cv2.imshow("Fire Detection", output)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to download the model from:")
        print("https://huggingface.co/kittendev/YOLOv8m-smoke-detection")
        print("And place it in the 'models' directory as 'fire_detection.pt'")