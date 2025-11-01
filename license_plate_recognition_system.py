# License Plate Recognition System — EasyOCR Edition (More Stable!)
# GPU-Enabled with EasyOCR instead of PaddleOCR
from __future__ import annotations

import cv2
import numpy as np
import torch
import easyocr
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

from ultralytics import YOLO
from database_manager import DatabaseManager

# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------

def expand_bbox(bbox: Tuple[int, int, int, int], pad_ratio: float,
                w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * pad_ratio), int(bh * pad_ratio)
    return (max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(w, x2 + pad_x), min(h, y2 + pad_y))

def preprocess_plate(img: np.ndarray) -> np.ndarray:
    """Enhanced preprocessing for license plate OCR"""
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Resize to make characters clearer
    height, width = gray.shape
    if width < 200:
        scale_factor = 200 / width
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=30)
    
    # Apply threshold
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------
class LicensePlateRecognitionSystem:
    """GPU-accelerated License Plate Recognition with EasyOCR"""

    def __init__(self,
                 db: DatabaseManager,
                 plate_detector_weights: str | Path = "best.pt",
                 ocr_lang: List[str] = ['en']):
        self.db = db

        self.db_manager = DatabaseManager(**self.db)
        self.db_manager.connect()

        # Check GPU availability
        use_gpu = torch.cuda.is_available()
        device = "cuda" if use_gpu else "cpu"
        
        print(f"🎮 License Plate System using: {device}")
        if use_gpu:
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

        plate_path = Path(plate_detector_weights)
        if not plate_path.exists():
            raise FileNotFoundError(
                f"Plate detector weights '{plate_path}' not found. "
                "Place 'best.pt' in the project root.")
        
        # Load YOLO on GPU
        print(f"Loading YOLO plate detector from: {plate_path}")
        self.plate_detector = YOLO(str(plate_path))
        if use_gpu:
            self.plate_detector.to('cuda')
        
        # Initialize EasyOCR with GPU support
        print(f"Initializing EasyOCR (GPU: {use_gpu})...")
        self.reader = easyocr.Reader(
            ocr_lang,
            gpu=use_gpu,
            verbose=False
        )
        
        print(f"✅ License plate system ready (GPU: {use_gpu})")

    # --------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Detects license plates, performs OCR, looks up owner,
        and returns detection results.
        """
        h, w = frame.shape[:2]
        outs = []

        # Run plate detection (GPU-accelerated)
        detection_results = self.plate_detector.predict(
            frame, 
            imgsz=640, 
            conf=0.25,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=False
        )[0].boxes
        
        for plate in detection_results:
            x1, y1, x2, y2 = map(int, plate.xyxy[0])
            x1, y1, x2, y2 = expand_bbox((x1, y1, x2, y2), 0.05, w, h)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Preprocess the cropped plate region
            pre = preprocess_plate(crop)

            # Perform OCR with EasyOCR (GPU-accelerated)
            try:
                results = self.reader.readtext(pre)
                
                text = ""
                conf = 0.0
                
                # Extract text and confidence from EasyOCR results
                for (bbox, detected_text, confidence) in results:
                    # Clean text
                    cleaned_text = detected_text.upper().replace(" ", "")
                    # Remove special characters except letters and numbers
                    cleaned_text = ''.join(c for c in cleaned_text if c.isalnum())
                    text += cleaned_text
                    conf = max(conf, confidence)
                
                # If no text detected, try original image
                if not text or conf < 0.5:
                    results = self.reader.readtext(crop)
                    for (bbox, detected_text, confidence) in results:
                        cleaned_text = detected_text.upper().replace(" ", "")
                        cleaned_text = ''.join(c for c in cleaned_text if c.isalnum())
                        text += cleaned_text
                        conf = max(conf, confidence)
                
            except Exception as e:
                print(f"OCR error: {e}")
                text = ""
                conf = 0.0

            # Clean up text
            text = text.strip()
            if text:
                print(f"Detected plate: {text} (confidence: {conf:.2f})")

            # Lookup plate owner and determine authorization
            owner = "Unknown car"
            authorised = False

            if conf > 0.5 and text:  # Lower threshold for EasyOCR
                owner_name = self.db_manager.lookup_owner_by_plate(text)
                if owner_name:
                    owner = owner_name
                    authorised = True

            # Collect result
            ts = datetime.now()
            plate_number = text if text else "UNKNOWN"
            outs.append({
                # GUI needs these ↓
                "timestamp": ts,
                "plate_number": plate_number,
                "vehicle_type": None,
                "is_authorized": authorised,

                # Original fields
                "bbox": (x1, y1, x2, y2),
                "plate": plate_number,
                "confidence": float(conf),
                "owner": owner,
                "authorised": authorised,
            })

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

    # --------------------------------------------------------------
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


# ------------------------------------------------------------------
# Quick demo (press Q to quit)
# ------------------------------------------------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    db_config = {
        'host': 'localhost',
        'database': 'facial_recognition',
        'user': 'postgres',
        'password': 'incorect'
    }
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