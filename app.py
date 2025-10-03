#!/usr/bin/env python3
"""
Optimized FastAPI Backend for Security System
With improved multi-threading for smooth video streaming
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import cv2
import asyncio
import json
import base64
import threading
import queue
import time
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import logging

from facial_recognition_system import FacialRecognitionSystem
from license_plate_recognition_system import LicensePlateRecognitionSystem
from database_manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class PersonRegistration(BaseModel):
    name: str
    employee_id: str
    department: Optional[str] = None
    authorized_zones: Optional[List[str]] = None

class LicensePlateRegistration(BaseModel):
    plate_number: str
    vehicle_type: Optional[str] = None
    owner_name: Optional[str] = None
    owner_id: Optional[str] = None
    is_authorized: bool = True
    notes: Optional[str] = None

class SystemConfig(BaseModel):
    recognition_threshold: Optional[float] = None
    min_confidence: Optional[float] = None
    camera_id: Optional[str] = None

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'facial_recognition_db',
    'user': 'postgres',
    'password': 'admin'
}

def convert_to_serializable(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj


class OptimizedSecuritySystemAPI:
    def __init__(self):
        self.app = FastAPI(title="Security System API", version="1.0.0")
        self.setup_cors()
        self.setup_routes()
        
        # Initialize systems
        self.db = DatabaseManager(**DB_CONFIG)
        self.db.connect()
        
        self.face_system = FacialRecognitionSystem(DB_CONFIG, camera_id="0")
        self.plate_system = LicensePlateRecognitionSystem(DB_CONFIG)
        
        # Video streaming
        self.video_capture = None
        self.is_streaming = False
        self.current_mode = "both"
        
        # Optimized queue system
        self.raw_frame_queue = queue.Queue(maxsize=2)  # Small queue for raw frames
        self.processed_frame_queue = queue.Queue(maxsize=2)  # Small queue for processed frames
        self.client_connections: List[WebSocket] = []
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Frame skip counter for performance
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_counter = 0
        
        # Processing flags
        self.processing_lock = threading.Lock()
        self.last_processed_frame = None
        self.last_results = {'face': [], 'plate': []}
        
        # Start background tasks
        self.setup_background_tasks()
        
    def setup_cors(self):
        """Setup CORS for React frontend"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {"message": "Security System API", "status": "online"}
        
        @self.app.get("/api/status")
        async def get_status():
            stats = self.db.get_statistics()
            return {
                "status": "online",
                "streaming": self.is_streaming,
                "mode": self.current_mode,
                "statistics": stats,
                "connected_clients": len(self.client_connections)
            }
        
        @self.app.post("/api/camera/start")
        async def start_camera(camera_id: int = 0):
            try:
                if not self.is_streaming:
                    self.video_capture = cv2.VideoCapture(camera_id)
                    if self.video_capture.isOpened():
                        # Set camera properties for better performance
                        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.video_capture.set(cv2.CAP_PROP_FPS, 30)
                        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        self.is_streaming = True
                        logger.info(f"✅ Camera {camera_id} started successfully")
                        return {"status": "success", "message": "Camera started"}
                    else:
                        raise HTTPException(status_code=400, detail="Could not open camera")
                else:
                    return {"status": "info", "message": "Camera already running"}
            except Exception as e:
                logger.error(f"❌ Error starting camera: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/camera/stop")
        async def stop_camera():
            try:
                if self.is_streaming:
                    self.is_streaming = False
                    if self.video_capture:
                        self.video_capture.release()
                    logger.info("🛑 Camera stopped")
                    return {"status": "success", "message": "Camera stopped"}
                else:
                    return {"status": "info", "message": "Camera not running"}
            except Exception as e:
                logger.error(f"❌ Error stopping camera: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/camera/mode")
        async def set_mode(request: dict):
            mode = request.get("mode")
            if mode not in ["face", "plate", "both"]:
                raise HTTPException(status_code=400, detail="Invalid mode")
            self.current_mode = mode
            logger.info(f"🔄 Detection mode set to: {mode}")
            return {"status": "success", "mode": mode}
        
        @self.app.post("/api/persons/register")
        async def register_person(person: PersonRegistration):
            try:
                return {
                    "status": "success", 
                    "message": f"Person {person.name} registered successfully",
                    "person": person.dict()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/plates/register")
        async def register_plate(plate: LicensePlateRegistration):
            try:
                success = self.plate_system.register_plate(
                    plate_number=plate.plate_number,
                    vehicle_type=plate.vehicle_type,
                    owner_name=plate.owner_name,
                    owner_id=plate.owner_id,
                    is_authorized=plate.is_authorized,
                    notes=plate.notes
                )
                
                if success:
                    return {
                        "status": "success",
                        "message": f"License plate {plate.plate_number} registered successfully"
                    }
                else:
                    raise HTTPException(status_code=400, detail="Failed to register plate")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/logs/vehicle")
        async def get_vehicle_logs(limit: int = 50):
            try:
                logs = self.db.get_vehicle_access_logs(limit=limit)
                return {"logs": convert_to_serializable(logs), "count": len(logs)}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/statistics")
        async def get_statistics():
            try:
                stats = self.db.get_statistics()
                return convert_to_serializable(stats)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.client_connections.append(websocket)
            logger.info(f"🔌 WebSocket client connected. Total: {len(self.client_connections)}")
            
            try:
                while True:
                    try:
                        # Get processed frame from queue (non-blocking)
                        message = self.processed_frame_queue.get(block=False)
                        await websocket.send_text(json.dumps(message))
                    except queue.Empty:
                        pass
                    except Exception as e:
                        logger.error(f"❌ Error sending WebSocket message: {e}")
                        break
                    
                    # Small delay to prevent CPU spinning
                    await asyncio.sleep(0.03)  # ~30 FPS
                    
            except WebSocketDisconnect:
                if websocket in self.client_connections:
                    self.client_connections.remove(websocket)
                logger.info(f"🔌 WebSocket disconnected. Total: {len(self.client_connections)}")
            except Exception as e:
                logger.error(f"❌ WebSocket error: {e}")
                if websocket in self.client_connections:
                    self.client_connections.remove(websocket)
    
    def setup_background_tasks(self):
        """Setup optimized background processing"""
        
        def capture_thread():
            """Thread 1: Capture frames from camera"""
            logger.info("🎥 Capture thread started")
            
            while True:
                try:
                    if self.is_streaming and self.video_capture and self.video_capture.isOpened():
                        ret, frame = self.video_capture.read()
                        if ret:
                            # Only add frame if queue is not full (drop frames if processing is slow)
                            try:
                                self.raw_frame_queue.put(frame, block=False)
                            except queue.Full:
                                # Drop frame if queue is full
                                pass
                    else:
                        time.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"❌ Error in capture thread: {e}")
                    time.sleep(0.1)
        
        def processing_thread():
            """Thread 2: Process frames with AI models"""
            logger.info("🧠 Processing thread started")
            
            while True:
                try:
                    # Get frame from capture queue
                    frame = self.raw_frame_queue.get(timeout=1.0)
                    
                    # Frame skipping for performance
                    self.frame_counter += 1
                    if self.frame_counter % self.frame_skip != 0:
                        # Still send raw frame to keep video smooth
                        self._encode_and_queue_frame(frame, [], [])
                        continue
                    
                    # Process frame based on mode
                    face_results = []
                    plate_results = []
                    processed_frame = frame.copy()
                    
                    mode = self.current_mode
                    
                    # Use thread pool for parallel processing
                    if mode in ["face", "both"]:
                        face_future = self.thread_pool.submit(self._process_faces, frame)
                    
                    if mode in ["plate", "both"]:
                        plate_future = self.thread_pool.submit(self._process_plates, frame)
                    
                    # Get results
                    if mode in ["face", "both"]:
                        try:
                            processed_frame, face_results = face_future.result(timeout=0.5)
                        except Exception as e:
                            logger.error(f"Face processing error: {e}")
                    
                    if mode in ["plate", "both"]:
                        try:
                            plate_results = plate_future.result(timeout=0.5)
                            if plate_results:
                                processed_frame = self.plate_system.draw_outputs(processed_frame, plate_results)
                        except Exception as e:
                            logger.error(f"Plate processing error: {e}")
                    
                    # Store results for reuse
                    with self.processing_lock:
                        self.last_results = {
                            'face': face_results,
                            'plate': plate_results
                        }
                    
                    # Encode and queue for sending
                    self._encode_and_queue_frame(processed_frame, face_results, plate_results)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"❌ Error in processing thread: {e}")
                    time.sleep(0.1)
        
        # Start threads
        capture_t = threading.Thread(target=capture_thread, daemon=True)
        capture_t.start()
        
        processing_t = threading.Thread(target=processing_thread, daemon=True)
        processing_t.start()
    
    def _process_faces(self, frame):
        """Process faces in separate thread"""
        try:
            return self.face_system.process_frame(frame)
        except Exception as e:
            logger.error(f"Face processing error: {e}")
            return frame, []
    
    def _process_plates(self, frame):
        """Process license plates in separate thread"""
        try:
            return self.plate_system.process_frame(frame)
        except Exception as e:
            logger.error(f"Plate processing error: {e}")
            return []
    
    def _encode_and_queue_frame(self, frame, face_results, plate_results):
        """Encode frame to JPEG and add to output queue"""
        try:
            # Resize frame for faster encoding and transmission
            height, width = frame.shape[:2]
            if width > 800:
                scale = 800 / width
                new_width = 800
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Encode with lower quality for speed
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare message
            message = {
                "type": "video_frame",
                "frame": frame_base64,
                "face_results": convert_to_serializable(face_results),
                "plate_results": convert_to_serializable(plate_results),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to queue (drop old frames if full)
            try:
                self.processed_frame_queue.put(message, block=False)
            except queue.Full:
                # Remove oldest frame and add new one
                try:
                    self.processed_frame_queue.get(block=False)
                    self.processed_frame_queue.put(message, block=False)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")


# Create the API instance
security_api = OptimizedSecuritySystemAPI()
app = security_api.app

if __name__ == "__main__":
    import uvicorn
    logger.info("🔒 Starting Optimized Security System API...")
    logger.info("📡 API: http://localhost:8000")
    logger.info("🔌 WebSocket: ws://localhost:8000/ws")
    logger.info("📚 Docs: http://localhost:8000/docs")
    
    try:
        uvicorn.run(
            "app:app",
            host="0.0.0.0", 
            port=8000, 
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down API server...")
    except Exception as e:
        logger.error(f"❌ Error: {e}")