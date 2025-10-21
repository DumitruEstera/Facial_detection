#!/usr/bin/env python3
"""
License Plate Recognition System - PaddleOCR Client Version
Calls external PaddleOCR microservice for better accuracy
"""

import cv2
import numpy as np
import torch
import requests
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from ultralytics import YOLO
from database_manager import DatabaseManager

def expand_bbox(bbox: Tuple[int, int, int, int], pad_ratio: float,
                w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * pad_ratio), int(bh * pad_ratio)
    return (max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(w, x2 + pad_x), min(h, y2 + pad_y))

class LicensePlateRecognitionSystem:
    """GPU-accelerated License Plate Recognition with PaddleOCR Microservice"""

    def __init__(self,
                 db: DatabaseManager,
                 plate_detector_weights: str = "best.pt",
                 ocr_service_url: str = "http://localhost:8001"):
        
        self.db = db
        self.db_manager = DatabaseManager(**self.db)
        self.db_manager.connect()

        # OCR Service URL
        self.ocr_service_url = ocr_service_url
        self._check_ocr_service()

        # Check GPU availability
        use_gpu = torch.cuda.is_available()
        device = "cuda" if use_gpu else "cpu"
        
        print(f"🎮 License Plate System using: {device}")
        if use_gpu:
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

        # Load YOLO detector
        plate_path = Path(plate_detector_weights)
        if not plate_path.exists():
            raise FileNotFoundError(f"Plate detector '{plate_path}' not found")
        
        print(f"Loading YOLO plate detector from: {plate_path}")
        self.plate_detector = YOLO(str(plate_path))
        if use_gpu:
            self.plate_detector.to('cuda')
        
        # Performance optimizations
        self.frame_skip_counter = 0
        self.frame_skip_rate = 10  # Process every 10th frame only!
        self.min_plate_width = 60   # LOWERED - accept smaller plates
        self.min_plate_height = 15  # LOWERED - accept smaller plates
        self.ocr_cache = {}
        self.cache_timeout = 5.0  # Longer cache (5 seconds)
        
        print(f"✅ License plate system ready (GPU: {use_gpu}, OCR: PaddleOCR)")
        print(f"⚡ FAST MODE: Skip every {self.frame_skip_rate} frames, Min: {self.min_plate_width}x{self.min_plate_height}px")

    def _check_ocr_service(self):
        """Check if OCR service is available"""
        try:
            response = requests.get(f"{self.ocr_service_url}/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ Connected to PaddleOCR service at {self.ocr_service_url}")
            else:
                print(f"⚠️  OCR service returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to OCR service at {self.ocr_service_url}")
            print(f"   Make sure to start: python ocr_service.py")
            raise ConnectionError(f"OCR service unavailable: {e}")

    def _call_ocr_service(self, image: np.ndarray, preprocess: bool = True) -> List[Dict]:
        """
        Call PaddleOCR microservice to perform OCR
        OPTIMIZED: Reduced timeout, better error handling
        """
        try:
            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Call OCR service with shorter timeout
            response = requests.post(
                f"{self.ocr_service_url}/ocr",
                json={
                    "image_base64": image_base64,
                    "preprocess": preprocess
                },
                timeout=2  # Reduced from 5 to 2 seconds
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"OCR service error: {response.status_code}")
                return []
                
        except requests.exceptions.Timeout:
            return []  # Silent timeout
        except Exception as e:
            return []  # Silent error
    
    def _clean_cache(self, current_time: datetime):
        """Clean expired cache entries"""
        expired_keys = [
            key for key, (timestamp, _, _) in self.ocr_cache.items()
            if (current_time - timestamp).total_seconds() > self.cache_timeout
        ]
        for key in expired_keys:
            del self.ocr_cache[key]

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect license plates - OPTIMIZED for performance
        """
        h, w = frame.shape[:2]
        outs = []
        current_time = datetime.now()

        # ALWAYS detect plates (fast GPU operation)
        detection_results = self.plate_detector.predict(
            frame, 
            imgsz=640, 
            conf=0.4,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=False
        )[0].boxes
        
        for plate in detection_results:
            x1, y1, x2, y2 = map(int, plate.xyxy[0])
            x1, y1, x2, y2 = expand_bbox((x1, y1, x2, y2), 0.1, w, h)
            
            plate_width = x2 - x1
            plate_height = y2 - y1
            
            # Skip tiny plates
            if plate_width < self.min_plate_width or plate_height < self.min_plate_height:
                continue
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Resize for better OCR
            if plate_width < 300:
                scale = 300 / plate_width
                new_width = 300
                new_height = int(plate_height * scale)
                crop = cv2.resize(crop, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            # Check cache first
            cache_key = f"{x1}_{y1}_{x2}_{y2}"
            text = ""
            conf = 0.0
            
            if cache_key in self.ocr_cache:
                cached_time, cached_text, cached_conf = self.ocr_cache[cache_key]
                if (current_time - cached_time).total_seconds() < self.cache_timeout:
                    text = cached_text
                    conf = cached_conf
            
            # PERFORMANCE: Only do OCR every 5th frame for this specific plate
            # Use position-based frame counter
            if not text:
                frame_count = getattr(self, f'_counter_{cache_key}', 0)
                self.__dict__[f'_counter_{cache_key}'] = frame_count + 1
                
                if frame_count % 5 == 0:  # Every 5th time we see this plate position
                    ocr_results = self._call_ocr_service(crop, preprocess=False)
                    
                    if ocr_results:
                        for result in ocr_results:
                            text += result['text']
                            conf = max(conf, result['confidence'])
                        
                        # Cache the result for 5 seconds
                        self.ocr_cache[cache_key] = (current_time, text, conf)
            
            # Always show the plate box with current info
            plate_number = text if text else "DETECTING..."
            owner = "Unknown car"
            authorised = False

            if conf > 0.5 and text:
                owner_name = self.db_manager.lookup_owner_by_plate(text)
                if owner_name:
                    owner = owner_name
                    authorised = True

            result = {
                "timestamp": current_time,
                "plate_number": plate_number,
                "vehicle_type": None,
                "is_authorized": authorised,
                "bbox": (x1, y1, x2, y2),
                "plate": plate_number,
                "confidence": float(conf),
                "owner": owner,
                "authorised": authorised,
            }
            
            outs.append(result)
        
        # Clean cache
        self._clean_cache(current_time)
        
        return outs
    
    def register_plate(self, plate_number: str, vehicle_type: str = None,
                       owner_name: str = None, owner_id: str = None,
                       is_authorized: bool = True, expiry_date: datetime = None,
                       notes: str = None) -> bool:
        """Register a new license plate in the database."""
        try:
            plate_id = self.db_manager.add_license_plate(
                plate_number=plate_number,
                vehicle_type=vehicle_type,
                owner_name=owner_name,
                owner_id=owner_id,
                is_authorized=is_authorized,
                expiry_date=expiry_date,
                notes=notes
            )
            print(f"Registered plate with ID: {plate_id}")
            return True
        except Exception as e:
            print(f"[!] Failed to register plate '{plate_number}': {e}")
            return False

    @staticmethod
    def draw_outputs(frame: np.ndarray, outs: List[Dict]) -> np.ndarray:
        """Draw detection boxes and labels on frame"""
        for o in outs:
            x1, y1, x2, y2 = o["bbox"]
            colour = (0, 255, 0) if o["authorised"] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            for i, line in enumerate([o["owner"], o["plate"],
                                      f"conf:{o['confidence']:.2f}"]):
                cv2.putText(frame, line, (x1, y2 + 22 + 20 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        return frame


if __name__ == "__main__":
    # Test the system
    cap = cv2.VideoCapture(0)
    db_config = {
        'host': 'localhost',
        'database': 'facial_recognition_db',
        'user': 'postgres',
        'password': 'admin'
    }
    
    print("Make sure OCR service is running: conda activate paddle_ocr_env && python ocr_service.py")
    lpr = LicensePlateRecognitionSystem(db_config)

    print("Press 'q' to quit")
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        detections = lpr.process_frame(frm)
        vis = LicensePlateRecognitionSystem.draw_outputs(frm.copy(), detections)
        cv2.imshow("LPR", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()