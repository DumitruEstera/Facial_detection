#!/usr/bin/env python3
"""
PaddleOCR Microservice - Runs in separate conda environment
Dedicated service for license plate OCR using PaddlePaddle-GPU
Port: 8001
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np
import base64
from paddleocr import PaddleOCR
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PaddleOCR Service", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize PaddleOCR with GPU (v3.2.0+ minimal config)
logger.info("🔥 Initializing PaddleOCR with GPU...")
try:
    ocr = PaddleOCR(lang='en', device='gpu')
    logger.info("✅ PaddleOCR ready on GPU!")
except Exception as e:
    logger.warning(f"⚠️  GPU initialization failed, trying CPU: {e}")
    ocr = PaddleOCR(lang='en', device='cpu')
    logger.info("✅ PaddleOCR ready on CPU!")

class OCRRequest(BaseModel):
    image_base64: str
    preprocess: bool = True

class OCRResult(BaseModel):
    text: str
    confidence: float
    bbox: List[List[int]]

@app.get("/")
async def root():
    return {
        "service": "PaddleOCR Microservice",
        "status": "online",
        "gpu_enabled": True
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "gpu": True}

@app.post("/ocr", response_model=List[OCRResult])
async def perform_ocr(request: OCRRequest):
    """
    Perform OCR on base64 encoded image
    """
    try:
        # Decode image
        image_bytes = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            print("❌ Invalid image!")
            raise HTTPException(status_code=400, detail="Invalid image")
        
        
        # Optional preprocessing
        if request.preprocess:
            image = preprocess_plate(image)
        
        # Perform OCR
        result = ocr.predict(image)
        
        # Parse results (v3.2.0 returns list with one dict)
        ocr_results = []
        
        if result and isinstance(result, list) and len(result) > 0:
            # Get the first item which contains all the OCR data
            item = result[0]
            
            if isinstance(item, dict):
                # Extract texts, scores, and polygons
                texts = item.get('rec_texts', [])
                scores = item.get('rec_scores', [])
                polys = item.get('rec_polys', [])
                
                
                # Process each detected text
                for i, text in enumerate(texts):
                    confidence = scores[i] if i < len(scores) else 0.0
                    poly = polys[i] if i < len(polys) else []
                    
                    # Convert polygon to bbox format [[x,y], ...]
                    bbox = []
                    if len(poly) > 0:
                        try:
                            bbox = [[int(p[0]), int(p[1])] for p in poly]
                        except:
                            bbox = []
                    
                    # Clean text
                    cleaned_text = str(text).upper().replace(" ", "")
                    cleaned_text = ''.join(c for c in cleaned_text if c.isalnum())
                    
                    if cleaned_text:
                        ocr_results.append(OCRResult(
                            text=cleaned_text,
                            confidence=float(confidence),
                            bbox=bbox
                        ))
        
        return ocr_results
        
    except Exception as e:
        print(f"❌ OCR error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def preprocess_plate(img: np.ndarray) -> np.ndarray:
    """Enhanced preprocessing for license plates"""
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Resize for better OCR
    height, width = gray.shape
    if width < 200:
        scale_factor = 200 / width
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=30)
    
    # Threshold
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to BGR for PaddleOCR
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

if __name__ == "__main__":
    logger.info("🚀 Starting PaddleOCR Microservice...")
    logger.info("📡 Service: http://localhost:8001")
    logger.info("🔥 GPU: Enabled")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )