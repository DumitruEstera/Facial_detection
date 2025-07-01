import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
import torch

class YOLOv8VehicleDetector:
    def __init__(self, model_path: str = None, device: str = 'auto'):
        """
        Initialize YOLOv8 vehicle and license plate detector
        
        Args:
            model_path: Path to custom YOLOv8 model (optional)
            device: Device to run inference on ('cpu', 'cuda', 'auto')
        """
        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load model
        if model_path:
            # Load custom model for license plate detection
            self.model = YOLO(model_path)
        else:
            # Use pretrained YOLOv8 model
            # For vehicles, we'll use the standard model
            # For license plates, you'll need a custom trained model
            self.model = YOLO('yolov8n.pt')  # nano model for speed
            
        # Move model to device
        self.model.to(self.device)
        
        # Vehicle classes in COCO dataset
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        # Confidence thresholds
        self.vehicle_conf_threshold = 0.5
        self.plate_conf_threshold = 0.7
        
    def detect_vehicles(self, image: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of dictionaries containing vehicle detections
        """
        # Run inference
        results = self.model(image, conf=self.vehicle_conf_threshold, device=self.device)
        
        detections = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Get box coordinates
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    
                    # Get class and confidence
                    cls = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    
                    # Check if it's a vehicle
                    if cls in self.vehicle_classes:
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'class': self.vehicle_classes[cls],
                            'confidence': conf
                        })
                        
        return detections
        
    def detect_license_plates(self, image: np.ndarray, vehicle_bbox: List[int] = None) -> List[Dict]:
        """
        Detect license plates in an image or within a vehicle bounding box
        
        Args:
            image: Input image (BGR format)
            vehicle_bbox: Optional vehicle bounding box to search within [x1, y1, x2, y2]
            
        Returns:
            List of dictionaries containing license plate detections
        """
        # If vehicle bbox is provided, crop the image
        if vehicle_bbox:
            x1, y1, x2, y2 = vehicle_bbox
            roi = image[y1:y2, x1:x2]
        else:
            roi = image
            x1, y1 = 0, 0
            
        # For license plate detection, you would use a custom trained model
        # This is a placeholder that simulates license plate detection
        # In production, train a YOLOv8 model specifically for license plates
        
        plate_detections = []
        
        # Placeholder: detect potential license plate regions
        # You should replace this with actual YOLOv8 license plate model inference
        height, width = roi.shape[:2]
        
        # Simulate detection in lower part of vehicle
        if vehicle_bbox:
            # License plates are typically in the lower 40% of the vehicle
            plate_region_y = int(height * 0.6)
            
            # Simple edge detection to find rectangular regions
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter based on aspect ratio and position
                aspect_ratio = w / h if h > 0 else 0
                
                if (2.0 < aspect_ratio < 6.0 and  # License plate aspect ratio
                    y > plate_region_y and  # In lower part
                    w > width * 0.2 and  # Minimum width
                    h > height * 0.05):  # Minimum height
                    
                    plate_detections.append({
                        'bbox': [x1 + x, y1 + y, x1 + x + w, y1 + y + h],
                        'confidence': 0.8,  # Placeholder confidence
                        'vehicle_bbox': vehicle_bbox
                    })
                    
        return plate_detections
        
    def detect_vehicles_and_plates(self, image: np.ndarray) -> Dict[str, List[Dict]]:
        """
        Detect both vehicles and their license plates
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with 'vehicles' and 'plates' lists
        """
        # Detect vehicles
        vehicles = self.detect_vehicles(image)
        
        # Detect license plates for each vehicle
        all_plates = []
        
        for vehicle in vehicles:
            # Search for license plates within each vehicle
            plates = self.detect_license_plates(image, vehicle['bbox'])
            
            # Add vehicle info to each plate
            for plate in plates:
                plate['vehicle_type'] = vehicle['class']
                plate['vehicle_confidence'] = vehicle['confidence']
                
            all_plates.extend(plates)
            
        return {
            'vehicles': vehicles,
            'plates': all_plates
        }
        
    def draw_detections(self, image: np.ndarray, detections: Dict[str, List[Dict]]) -> np.ndarray:
        """
        Draw detection boxes on image
        
        Args:
            image: Input image
            detections: Dictionary with vehicles and plates
            
        Returns:
            Image with drawn detections
        """
        image_copy = image.copy()
        
        # Draw vehicles
        for vehicle in detections.get('vehicles', []):
            x1, y1, x2, y2 = vehicle['bbox']
            
            # Draw rectangle
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{vehicle['class']} ({vehicle['confidence']:.2f})"
            cv2.putText(image_copy, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Draw license plates
        for plate in detections.get('plates', []):
            x1, y1, x2, y2 = plate['bbox']
            
            # Draw rectangle
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Add label
            label = f"Plate ({plate['confidence']:.2f})"
            cv2.putText(image_copy, label, (x1, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        return image_copy
        
    def extract_plate_image(self, image: np.ndarray, plate_bbox: List[int],
                           padding: int = 5) -> Optional[np.ndarray]:
        """
        Extract license plate region from image
        
        Args:
            image: Input image
            plate_bbox: Plate bounding box [x1, y1, x2, y2]
            padding: Padding around the plate
            
        Returns:
            Cropped license plate image or None
        """
        x1, y1, x2, y2 = plate_bbox
        height, width = image.shape[:2]
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        # Extract plate
        plate_image = image[y1:y2, x1:x2]
        
        if plate_image.size == 0:
            return None
            
        return plate_image


class LicensePlateYOLOTrainer:
    """
    Helper class to train YOLOv8 for license plate detection
    This is provided as reference for training your own model
    """
    
    @staticmethod
    def prepare_dataset_structure():
        """
        Creates the required directory structure for YOLOv8 training
        """
        import os
        
        dirs = [
            'license_plate_dataset/train/images',
            'license_plate_dataset/train/labels',
            'license_plate_dataset/val/images',
            'license_plate_dataset/val/labels',
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
        # Create data.yaml file
        data_yaml = """
train: ./license_plate_dataset/train/images
val: ./license_plate_dataset/val/images

nc: 1  # number of classes
names: ['license_plate']  # class names
"""
        
        with open('license_plate_dataset/data.yaml', 'w') as f:
            f.write(data_yaml)
            
        print("Dataset structure created. Add your annotated images and labels.")
        
    @staticmethod
    def train_model(data_yaml_path: str, epochs: int = 100):
        """
        Train YOLOv8 model for license plate detection
        
        Args:
            data_yaml_path: Path to data.yaml file
            epochs: Number of training epochs
        """
        # Load a base model
        model = YOLO('yolov8n.pt')
        
        # Train the model
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=16,
            name='license_plate_detector'
        )
        
        return results