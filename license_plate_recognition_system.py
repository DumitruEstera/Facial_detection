import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime
import threading
import queue
import os

# Import our modules
from database_manager import DatabaseManager
from vehicle_detection import YOLOv8VehicleDetector
from license_plate_ocr import LicensePlateOCR

class LicensePlateRecognitionSystem:
    def __init__(self, db_config: Dict, camera_id: str = "0", 
                 save_images: bool = True, image_dir: str = "vehicle_captures"):
        """
        Initialize the license plate recognition system
        
        Args:
            db_config: Database configuration dictionary
            camera_id: Camera identifier
            save_images: Whether to save captured vehicle images
            image_dir: Directory to save vehicle images
        """
        self.camera_id = camera_id
        self.save_images = save_images
        self.image_dir = image_dir
        
        # Create image directory if needed
        if self.save_images:
            os.makedirs(self.image_dir, exist_ok=True)
            
        # Initialize components
        print("Initializing database connection...")
        self.db = DatabaseManager(**db_config)
        self.db.connect()
        
        print("Initializing vehicle detector...")
        self.vehicle_detector = YOLOv8VehicleDetector()
        
        print("Initializing license plate OCR...")
        self.plate_ocr = LicensePlateOCR(languages=['en'])
        
        # Cache for authorized plates
        self.authorized_plates = set(self.db.get_all_authorized_plates())
        self.cache_update_time = time.time()
        self.cache_ttl = 300  # 5 minutes
        
        # Detection parameters
        self.min_plate_confidence = 0.7
        self.min_ocr_confidence = 0.6
        
        # Tracking to avoid duplicate logs
        self.recent_detections = {}  # plate -> last detection time
        self.detection_cooldown = 30  # seconds
        
        # Threading for performance
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.is_running = False
        
    def update_authorized_plates_cache(self):
        """Update the cache of authorized plates"""
        current_time = time.time()
        if current_time - self.cache_update_time > self.cache_ttl:
            self.authorized_plates = set(self.db.get_all_authorized_plates())
            self.cache_update_time = current_time
            print(f"Updated authorized plates cache: {len(self.authorized_plates)} plates")
            
    def register_plate(self, plate_number: str, vehicle_type: str = None,
                      owner_name: str = None, owner_id: str = None,
                      is_authorized: bool = True, notes: str = None) -> bool:
        """
        Register a new license plate in the system
        
        Args:
            plate_number: License plate number
            vehicle_type: Type of vehicle
            owner_name: Owner's name
            owner_id: Owner's ID/employee ID
            is_authorized: Whether the plate is authorized
            notes: Additional notes
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            plate_id = self.db.add_license_plate(
                plate_number=plate_number,
                vehicle_type=vehicle_type,
                owner_name=owner_name,
                owner_id=owner_id,
                is_authorized=is_authorized,
                notes=notes
            )
            
            # Update cache if authorized
            if is_authorized:
                self.authorized_plates.add(plate_number)
                
            print(f"Successfully registered plate: {plate_number}")
            return True
            
        except Exception as e:
            print(f"Error registering plate: {e}")
            return False
            
    def is_plate_authorized(self, plate_number: str) -> bool:
        """Check if a plate is authorized"""
        self.update_authorized_plates_cache()
        return plate_number in self.authorized_plates
        
    def should_log_detection(self, plate_number: str) -> bool:
        """Check if we should log this detection (avoid duplicates)"""
        current_time = time.time()
        
        if plate_number in self.recent_detections:
            last_time = self.recent_detections[plate_number]
            if current_time - last_time < self.detection_cooldown:
                return False
                
        self.recent_detections[plate_number] = current_time
        return True
        
    def save_vehicle_image(self, image: np.ndarray, plate_number: str, 
                          is_authorized: bool) -> Optional[str]:
        """Save vehicle image to disk"""
        if not self.save_images:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "authorized" if is_authorized else "unauthorized"
        filename = f"{timestamp}_{plate_number}_{status}.jpg"
        filepath = os.path.join(self.image_dir, filename)
        
        try:
            cv2.imwrite(filepath, image)
            return filepath
        except Exception as e:
            print(f"Error saving image: {e}")
            return None
            
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame for vehicle and license plate detection
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Annotated frame and list of detection results
        """
        # Detect vehicles and plates
        detections = self.vehicle_detector.detect_vehicles_and_plates(frame)
        
        recognition_results = []
        
        # Process each detected license plate
        for plate_detection in detections.get('plates', []):
            # Extract plate image
            plate_image = self.vehicle_detector.extract_plate_image(
                frame, plate_detection['bbox']
            )
            
            if plate_image is None:
                continue
                
            # Run OCR on the plate
            ocr_result = self.plate_ocr.read_plate(plate_image)
            
            if (ocr_result['valid'] and 
                ocr_result['confidence'] >= self.min_ocr_confidence):
                
                plate_number = ocr_result['text']
                
                # Check authorization
                is_authorized = self.is_plate_authorized(plate_number)
                
                # Create result
                result = {
                    'plate_number': plate_number,
                    'confidence': ocr_result['confidence'],
                    'is_authorized': is_authorized,
                    'vehicle_type': plate_detection.get('vehicle_type', 'unknown'),
                    'bbox': plate_detection['bbox'],
                    'timestamp': datetime.now()
                }
                
                recognition_results.append(result)
                
                # Log to database if not recently logged
                if self.should_log_detection(plate_number):
                    # Save image if configured
                    image_path = self.save_vehicle_image(frame, plate_number, is_authorized)
                    
                    # Log to database
                    self.db.log_vehicle_access(
                        plate_number=plate_number,
                        camera_id=self.camera_id,
                        confidence=ocr_result['confidence'],
                        vehicle_type=plate_detection.get('vehicle_type'),
                        image_path=image_path
                    )
                    
        # Draw detections on frame
        annotated_frame = self.draw_results(frame, detections, recognition_results)
        
        return annotated_frame, recognition_results
        
    def draw_results(self, frame: np.ndarray, detections: Dict, 
                    recognition_results: List[Dict]) -> np.ndarray:
        """Draw detection and recognition results on frame"""
        frame_copy = frame.copy()
        
        # Draw vehicles
        for vehicle in detections.get('vehicles', []):
            x1, y1, x2, y2 = vehicle['bbox']
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{vehicle['class']} ({vehicle['confidence']:.2f})"
            cv2.putText(frame_copy, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Draw recognized plates
        for result in recognition_results:
            x1, y1, x2, y2 = result['bbox']
            
            # Color based on authorization
            color = (0, 255, 0) if result['is_authorized'] else (0, 0, 255)
            
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw plate number and status
            label = f"{result['plate_number']} ({result['confidence']:.2f})"
            status = "AUTHORIZED" if result['is_authorized'] else "UNAUTHORIZED"
            
            cv2.putText(frame_copy, label, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame_copy, status, (x1, y2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
        return frame_copy
        
    def start_video_stream(self, source: int = 0):
        """
        Start processing video stream from camera
        
        Args:
            source: Video source (0 for default camera, or video file path)
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
            
        print("Starting video stream... Press 'q' to quit")
        
        # FPS calculation
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            annotated_frame, results = self.process_frame(frame)
            
            # Calculate FPS
            fps_frame_count += 1
            if fps_frame_count >= 30:
                fps_end_time = time.time()
                fps = fps_frame_count / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                fps_frame_count = 0
                
            # Display FPS
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display statistics
            stats_text = f"Authorized Plates: {len(self.authorized_plates)}"
            cv2.putText(annotated_frame, stats_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display results
            cv2.imshow("License Plate Recognition System", annotated_frame)
            
            # Print recognition results
            for result in results:
                status = "AUTHORIZED" if result['is_authorized'] else "UNAUTHORIZED"
                print(f"Detected: {result['plate_number']} - {status} "
                      f"(Confidence: {result['confidence']:.2f})")
                
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    def test_on_image(self, image_path: str):
        """Test the system on a single image"""
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return
            
        # Process image
        annotated_image, results = self.process_frame(image)
        
        # Display results
        print(f"\nResults for {image_path}:")
        for result in results:
            status = "AUTHORIZED" if result['is_authorized'] else "UNAUTHORIZED"
            print(f"  Plate: {result['plate_number']} - {status}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Vehicle Type: {result['vehicle_type']}")
            print()
            
        # Show image
        cv2.imshow("License Plate Detection", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        db_stats = self.db.get_statistics()
        
        stats = {
            **db_stats,
            'cached_authorized_plates': len(self.authorized_plates),
            'recent_detections': len(self.recent_detections)
        }
        
        return stats
        
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        self.db.disconnect()
        print("License plate recognition system cleaned up")