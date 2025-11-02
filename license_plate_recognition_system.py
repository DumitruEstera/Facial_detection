#!/usr/bin/env python3
"""
License Plate Recognition System - FIXED FOR SMALL PLATES

This version automatically handles small plate crops by:
1. Increasing detection padding (0.05 → 0.20)
2. Upscaling crops to 250+ pixels wide
3. Using better interpolation

Works with EasyOCR (or switch to PaddleOCR for even better results)
"""

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


# ============================================================================
# HELPER UTILITIES - FIXED FOR SMALL PLATES
# ============================================================================

def expand_bbox(bbox: Tuple[int, int, int, int], pad_ratio: float,
                w: int, h: int) -> Tuple[int, int, int, int]:
    """Expand bounding box with padding"""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * pad_ratio), int(bh * pad_ratio)
    return (max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(w, x2 + pad_x), min(h, y2 + pad_y))


def upscale_if_small(crop: np.ndarray, 
                     target_width: int = 250,
                     max_scale: float = 3.0) -> np.ndarray:
    """
    Upscale crop if too small for reliable OCR
    
    CRITICAL FIX: Your plates are 116-151px wide (TOO SMALL)
    This upscales them to 250px for much better OCR results
    
    Args:
        crop: Original plate crop
        target_width: Target width in pixels (250 is good for OCR)
        max_scale: Maximum upscaling factor (3.0 = 3x max)
    
    Returns:
        Upscaled crop
    """
    if crop.size == 0:
        return crop
    
    h, w = crop.shape[:2]
    
    if w < target_width:
        # Calculate scale factor
        scale = target_width / w
        scale = min(scale, max_scale)  # Don't upscale too much
        
        # New dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Upscale with high-quality interpolation
        upscaled = cv2.resize(crop, (new_w, new_h), 
                             interpolation=cv2.INTER_CUBIC)
        
        print(f"📏 Upscaled plate: {w}×{h} → {new_w}×{new_h} (scale: {scale:.2f}x)")
        return upscaled
    
    return crop


def preprocess_plate(img: np.ndarray) -> np.ndarray:
    """
    Minimal preprocessing for OCR
    Now with proper handling for upscaled images
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Note: No additional resizing here - already done by upscale_if_small
    
    return gray


# ============================================================================
# MAIN SYSTEM - FIXED VERSION
# ============================================================================

class LicensePlateRecognitionSystem:
    """
    License Plate Recognition with SMALL PLATE FIX
    
    Key changes from original:
    1. Increased padding: 0.05 → 0.20 (captures more context)
    2. Automatic upscaling: crops upscaled to 250+ pixels
    3. Better interpolation: INTER_CUBIC for quality
    """

    def __init__(self,
                 db: DatabaseManager,
                 plate_detector_weights: str | Path = "best.pt",
                 ocr_lang: List[str] = ['en'],
                 use_gpu: bool = True,
                 target_plate_width: int = 250):
        """
        Initialize with small plate fixes
        
        Args:
            db: Database config dict
            plate_detector_weights: Path to YOLO weights
            ocr_lang: OCR languages
            use_gpu: Use GPU acceleration
            target_plate_width: Target width for upscaling (250 recommended)
        """
        self.db = db
        self.target_plate_width = target_plate_width

        self.db_manager = DatabaseManager(**self.db)
        self.db_manager.connect()

        # Check GPU
        use_gpu = torch.cuda.is_available() and use_gpu
        device = "cuda" if use_gpu else "cpu"
        
        print(f"🎮 System using: {device}")
        if use_gpu:
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

        # Load YOLO
        plate_path = Path(plate_detector_weights)
        if not plate_path.exists():
            raise FileNotFoundError(f"Weights not found: {plate_path}")
        
        print(f"Loading YOLO from: {plate_path}")
        self.plate_detector = YOLO(str(plate_path))
        if use_gpu:
            self.plate_detector.to('cuda')
        
        # Initialize EasyOCR
        print(f"Initializing EasyOCR (GPU: {use_gpu})...")
        self.reader = easyocr.Reader(
            ocr_lang,
            gpu=use_gpu,
            verbose=False
        )
        
        print("✅ System ready with SMALL PLATE FIX")
        print(f"   📏 Will upscale plates to {target_plate_width}+ pixels")

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process frame with SMALL PLATE FIXES
        """
        h, w = frame.shape[:2]
        outs = []

        # Detect plates
        detection_results = self.plate_detector.predict(
            frame, 
            imgsz=640, 
            conf=0.25,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=False
        )[0].boxes
        
        for plate in detection_results:
            x1, y1, x2, y2 = map(int, plate.xyxy[0])
            
            # FIX 1: INCREASE PADDING (0.05 → 0.20)
            x1, y1, x2, y2 = expand_bbox((x1, y1, x2, y2), 0.20, w, h)
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # FIX 2: UPSCALE IF TOO SMALL
            crop = upscale_if_small(crop, target_width=self.target_plate_width)
            
            # Preprocess
            pre = preprocess_plate(crop)

            # OCR
            text = ""
            conf = 0.0
            
            try:
                results = self.reader.readtext(pre)
                
                for (bbox, detected_text, confidence) in results:
                    cleaned = detected_text.upper().replace(" ", "")
                    cleaned = ''.join(c for c in cleaned if c.isalnum())
                    text += cleaned
                    conf = max(conf, confidence)
                
                # Fallback: try original crop if low confidence
                if (not text or conf < 0.4):
                    results_color = self.reader.readtext(crop)
                    text_color = ""
                    conf_color = 0.0
                    
                    for (bbox, detected_text, confidence) in results_color:
                        cleaned = detected_text.upper().replace(" ", "")
                        cleaned = ''.join(c for c in cleaned if c.isalnum())
                        text_color += cleaned
                        conf_color = max(conf_color, confidence)
                    
                    if conf_color > conf:
                        text = text_color
                        conf = conf_color
                
            except Exception as e:
                print(f"OCR error: {e}")
                text = ""
                conf = 0.0

            text = text.strip()
            if text and conf > 0.4:
                print(f"✅ Detected: {text} (conf: {conf:.2f})")

            # Lookup
            owner = "Unknown car"
            authorised = False

            if conf > 0.5 and text:
                owner_name = self.db_manager.lookup_owner_by_plate(text)
                if owner_name:
                    owner = owner_name
                    authorised = True

            # Result
            ts = datetime.now()
            plate_number = text if text else "UNKNOWN"
            
            outs.append({
                "timestamp": ts,
                "plate_number": plate_number,
                "vehicle_type": None,
                "is_authorized": authorised,
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
        """Register new plate"""
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
            print(f"Registered plate ID: {plate_id}")
            return True
        except Exception as e:
            print(f"Failed to register: {e}")
            return False

    @staticmethod
    def draw_outputs(frame: np.ndarray, outs: List[Dict]) -> np.ndarray:
        """Draw detections"""
        for o in outs:
            x1, y1, x2, y2 = o["bbox"]
            colour = (0, 255, 0) if o["authorised"] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            
            for i, line in enumerate([o["owner"], o["plate"],
                                      f"conf:{o['confidence']:.2f}"]):
                cv2.putText(frame, line, (x1, y2 + 22 + 20 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        return frame


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LICENSE PLATE RECOGNITION - FIXED FOR SMALL PLATES")
    print("="*70)
    print("\n✅ Fixes applied:")
    print("   1. Increased padding: 0.05 → 0.20 (more context)")
    print("   2. Auto-upscaling: plates scaled to 250+ pixels")
    print("   3. Better interpolation: INTER_CUBIC quality")
    print("\n📊 Expected improvement:")
    print("   Before: 116×33 pixels → OCR fails")
    print("   After:  250×70 pixels → OCR succeeds!")
    print("="*70 + "\n")
    
    db_config = {
        'host': 'localhost',
        'database': 'facial_recognition',
        'user': 'postgres',
        'password': 'incorect'
    }
    
    lpr = LicensePlateRecognitionSystem(db_config)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open camera")
        exit(1)
    
    print("✅ Camera opened")
    print("\nPress 'Q' to quit\n")
    
    frame_count = 0
    
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 2nd frame for performance
        if frame_count % 2 == 0:
            detections = lpr.process_frame(frm)
            vis = LicensePlateRecognitionSystem.draw_outputs(frm.copy(), detections)
        else:
            vis = frm
        
        cv2.putText(vis, "FIXED VERSION - Upscaling enabled", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("License Plate Recognition (FIXED) - Press Q to quit", vis)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()