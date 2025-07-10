# License‑Plate Recognition System — **Kaggle best.pt edition**
# ================================================================
# This version swaps in the YOLOv8 licence‑plate detector you downloaded
# from Kaggle (harshitsingh09/yolov8-license-plate-detector).  After
# unzipping, you should have a weight file called **best.pt**.  Place
# that file in the project root (or pass another path when instantiating
# the class) and you’re good to go.
#
# → Runtime deps (if missing)
#     pip install ultralytics paddleocr opencv-python-headless numpy
#
# --------------------------------------------------------------------
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from ultralytics import YOLO            # YOLOv8 engine
from paddleocr import PaddleOCR         # PP‑LPR recogniser
from datetime import datetime

from database_manager import DatabaseManager
from vehicle_detection import YOLOv8VehicleDetector


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
    up = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    eq = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    den = cv2.bilateralFilter(eq, 9, 75, 75)
    bin_img = cv2.adaptiveThreshold(den, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 5)
    return bin_img


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------
class LicensePlateRecognitionSystem:
    """Car → plate → OCR → DB lookup → overlay"""

    def __init__(self,
                 db: DatabaseManager,
                 vehicle_detector_weights: str | Path = "yolov8s.pt",
                 plate_detector_weights: str | Path = "best.pt",
                 use_cuda: bool = True,
                 ocr_lang: str = "en"):
        self.db = db
        self.car_detector = YOLOv8VehicleDetector(vehicle_detector_weights)

        plate_path = Path(plate_detector_weights)
        if not plate_path.exists():
            raise FileNotFoundError(
                f"Plate‑detector weights '{plate_path}' not found. "
                "Place 'best.pt' in the project root or pass its path "
                "to LicensePlateRecognitionSystem(..., plate_detector_weights='path').")
        self.plate_detector = YOLO(str(plate_path))

        self.ocr = PaddleOCR(lang=ocr_lang, use_angle_cls=False)

    # --------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Detects license plates in the given frame, performs OCR, looks up owner,
        and returns a list of result dicts containing bbox, plate text, confidence,
        owner, and authorization status.
        """
        h, w = frame.shape[:2]
        outs = []

        # Run plate detection
        detection_results = self.plate_detector.predict(frame, imgsz=640, conf=0.25)[0].boxes
        for plate in detection_results:
            x1, y1, x2, y2 = map(int, plate.xyxy[0])
            x1, y1, x2, y2 = expand_bbox((x1, y1, x2, y2), 0.05, w, h)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Preprocess the cropped plate region (e.g., binarization)
            pre = preprocess_plate(crop)
            # Ensure a 3-channel BGR image for PaddleOCR
            if pre.ndim == 2:
                pre = cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR)
            elif pre.ndim == 3 and pre.shape[2] == 1:
                pre = cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR)

            # Perform OCR on the preprocessed image
            rec_results = self.ocr.ocr(pre)
            text = ""
            conf = 0.0
            if rec_results:
                first = rec_results[0]
                # PaddleOCR v3+ returns dicts; older versions return [bbox, [text, conf]]
                if isinstance(first, dict):
                    text = first.get('text', '')
                    conf = first.get('confidence', 0.0)
                elif isinstance(first, (list, tuple)) and len(first) >= 2 and isinstance(first[1], (list, tuple)):
                    text, conf = first[1]
                # normalize text
                text = (text or "").upper().replace(" ", "")

            # Lookup plate owner and determine authorization
            # If db is a dict, use get; otherwise, call lookup method
            if isinstance(self.db, dict):
                owner = self.db.get(text)
            else:
                owner = self.db.lookup_owner_by_plate(text)
            authorised = bool(owner and text)

            # Collect result
            ts = datetime.now()
            plate_number = text if text else "UNKNOWN"
            outs.append({
                # GUI needs these ↓
                "timestamp": ts,
                "plate_number": plate_number,
                "vehicle_type": None,          # fill in later if you have it
                "is_authorized": authorised,

                # original fields (keep them if other code uses them)
                "bbox": (x1, y1, x2, y2),
                "plate": plate_number,
                "confidence": float(conf),
                "owner": owner if authorised else "Unknown car",
                "authorised": authorised,
            })

        return outs

    # --------------------------------------------------------------
    @staticmethod
    def draw_outputs(frame: np.ndarray, outs: List[Dict]) -> np.ndarray:
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
# Quick demo (press Q to quit)
# ------------------------------------------------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    db = DatabaseManager("sqlite:///security.db")
    lpr = LicensePlateRecognitionSystem(db)

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
