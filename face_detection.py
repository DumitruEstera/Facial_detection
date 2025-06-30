import cv2
import numpy as np
from typing import List, Tuple, Optional
import urllib.request
import os

class YuNetFaceDetector:
    def __init__(self, model_path: str = None, conf_threshold: float = 0.7, 
                 nms_threshold: float = 0.3, top_k: int = 5000):
        """
        Initialize YuNet face detector
        
        Args:
            model_path: Path to YuNet model file
            conf_threshold: Confidence threshold for face detection
            nms_threshold: NMS threshold for removing duplicate detections
            top_k: Keep top K detections before NMS
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        
        # Download model if not provided
        if model_path is None:
            model_path = self._download_model()
            
        # Initialize YuNet detector
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            self.conf_threshold,
            self.nms_threshold,
            self.top_k
        )
        
    def _download_model(self) -> str:
        """Download YuNet model if not exists"""
        model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        model_path = "face_detection_yunet_2023mar.onnx"
        
        if not os.path.exists(model_path):
            print("Downloading YuNet model...")
            urllib.request.urlretrieve(model_url, model_path)
            print("Model downloaded successfully")
            
        return model_path
        
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of face bounding boxes (x, y, width, height)
        """
        height, width = image.shape[:2]
        
        # Set input size
        self.detector.setInputSize((width, height))
        
        # Detect faces
        _, faces = self.detector.detect(image)
        
        face_boxes = []
        if faces is not None:
            for face in faces:
                # Extract bounding box
                x, y, w, h = face[:4].astype(int)
                
                # Ensure bounding box is within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)
                
                if w > 0 and h > 0:
                    face_boxes.append((x, y, w, h))
                    
        return face_boxes
        
    def extract_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                     margin: float = 0.2) -> Optional[np.ndarray]:
        """
        Extract face region from image with margin
        
        Args:
            image: Input image
            bbox: Face bounding box (x, y, width, height)
            margin: Margin to add around face (percentage of face size)
            
        Returns:
            Cropped face image or None if extraction fails
        """
        x, y, w, h = bbox
        height, width = image.shape[:2]
        
        # Add margin
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        # Calculate new coordinates with margin
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(width, x + w + margin_x)
        y2 = min(height, y + h + margin_y)
        
        # Extract face
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
            
        return face
        
    def draw_faces(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                   labels: List[str] = None, confidences: List[float] = None) -> np.ndarray:
        """
        Draw face bounding boxes on image
        
        Args:
            image: Input image
            faces: List of face bounding boxes
            labels: Optional labels for each face
            confidences: Optional confidence scores for each face
            
        Returns:
            Image with drawn bounding boxes
        """
        image_copy = image.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            # Draw rectangle
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label if provided
            if labels and i < len(labels):
                label = labels[i]
                if confidences and i < len(confidences):
                    label = f"{label} ({confidences[i]:.2f})"
                    
                # Calculate text size
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Draw background rectangle for text
                cv2.rectangle(
                    image_copy, 
                    (x, y - text_height - 4), 
                    (x + text_width, y), 
                    (0, 255, 0), 
                    -1
                )
                
                # Draw text
                cv2.putText(
                    image_copy, 
                    label, 
                    (x, y - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 0, 0), 
                    1
                )
                
        return image_copy