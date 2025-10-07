#!/usr/bin/env python3
"""
Fixed FastAPI Backend for Security System
Key fixes:
1. Better queue coordination - frames only added to raw queue if successfully queued for processing
2. Smarter merging logic - uses partial results if available
3. Increased timeout for result matching
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
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
        self.app = FastAPI(title="Security System API", version="2.0.1")
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
        
        # Multi-threaded queue system
        self.raw_frame_queue = queue.Queue(maxsize=10)
        self.face_processing_queue = queue.Queue(maxsize=10)
        self.plate_processing_queue = queue.Queue(maxsize=10)
        self.face_results_queue = queue.Queue(maxsize=10)
        self.plate_results_queue = queue.Queue(maxsize=10)
        self.processed_frame_queue = queue.Queue(maxsize=10)
        
        self.client_connections: List[WebSocket] = []
        
        # Frame management
        self.frame_skip = 1
        self.frame_counter = 0
        self.frame_id = 0
        
        # Statistics
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'face_detections': 0,
            'plate_detections': 0,
            'frames_dropped': 0,
            'queue_skips': 0
        }
        
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
            return {"message": "Security System API v2.0.1 (Fixed)", "status": "online"}
        
        @self.app.get("/api/status")
        async def get_status():
            stats = self.db.get_statistics()
            return {
                "status": "online",
                "streaming": self.is_streaming,
                "mode": self.current_mode,
                "statistics": stats,
                "connected_clients": len(self.client_connections),
                "performance": {
                    "frames_captured": self.stats['frames_captured'],
                    "frames_processed": self.stats['frames_processed'],
                    "face_detections": self.stats['face_detections'],
                    "plate_detections": self.stats['plate_detections'],
                    "frames_dropped": self.stats['frames_dropped'],
                    "queue_skips": self.stats['queue_skips']
                }
            }
        
        @self.app.post("/api/camera/start")
        async def start_camera(camera_id: int = 0):
            try:
                if not self.is_streaming:
                    self.video_capture = cv2.VideoCapture(camera_id)
                    if self.video_capture.isOpened():
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
                    self._clear_all_queues()
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
                        message = self.processed_frame_queue.get(block=False)
                        await websocket.send_text(json.dumps(message))
                    except queue.Empty:
                        pass
                    except Exception as e:
                        logger.error(f"❌ Error sending WebSocket message: {e}")
                        break
                    
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
        """Setup optimized multi-threaded processing pipeline"""
        
        # Thread 1: Video Capture - FIXED VERSION
        def capture_thread():
            """Capture frames and ensure they're queued for processing before adding to raw queue"""
            logger.info("🎥 Thread 1: Video Capture started (FIXED)")
            
            while True:
                try:
                    if self.is_streaming and self.video_capture and self.video_capture.isOpened():
                        ret, frame = self.video_capture.read()
                        if ret:
                            self.stats['frames_captured'] += 1
                            current_frame_id = self.frame_id
                            self.frame_id += 1
                            
                            # Frame skipping logic
                            self.frame_counter += 1
                            if self.frame_counter % self.frame_skip != 0:
                                continue
                            
                            mode = self.current_mode
                            
                            # FIX: Track if frame was successfully queued for processing
                            queued_for_face = False
                            queued_for_plate = False
                            
                            # Try to queue for face processing
                            if mode in ["face", "both"]:
                                try:
                                    self.face_processing_queue.put((current_frame_id, frame.copy()), block=False)
                                    queued_for_face = True
                                except queue.Full:
                                    self.stats['queue_skips'] += 1
                            else:
                                queued_for_face = True  # Not needed for this mode
                            
                            # Try to queue for plate processing
                            if mode in ["plate", "both"]:
                                try:
                                    self.plate_processing_queue.put((current_frame_id, frame.copy()), block=False)
                                    queued_for_plate = True
                                except queue.Full:
                                    self.stats['queue_skips'] += 1
                            else:
                                queued_for_plate = True  # Not needed for this mode
                            
                            # FIX: Only add to raw queue if successfully queued for processing
                            # OR if queues don't need processing for current mode
                            if queued_for_face and queued_for_plate:
                                try:
                                    self.raw_frame_queue.put((current_frame_id, frame.copy()), block=False)
                                except queue.Full:
                                    self.stats['frames_dropped'] += 1
                            else:
                                # Frame couldn't be queued properly, skip it
                                self.stats['frames_dropped'] += 1
                        else:
                            logger.warning("⚠️ Failed to read frame from camera")
                            time.sleep(0.1)
                    else:
                        time.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"❌ Error in capture thread: {e}")
                    time.sleep(0.1)
        
        # Thread 2: Facial Recognition Processing
        def face_processing_thread():
            """Process frames for facial recognition"""
            logger.info("👤 Thread 2: Face Recognition started")
            
            while True:
                try:
                    frame_id, frame = self.face_processing_queue.get(timeout=1.0)
                    
                    start_time = time.time()
                    processed_frame, face_results = self._process_faces(frame)
                    processing_time = time.time() - start_time
                    
                    if face_results:
                        self.stats['face_detections'] += len(face_results)
                        logger.debug(f"👤 Detected {len(face_results)} faces in {processing_time:.3f}s")
                    
                    try:
                        self.face_results_queue.put((frame_id, processed_frame, face_results), block=False)
                    except queue.Full:
                        try:
                            self.face_results_queue.get(block=False)
                            self.face_results_queue.put((frame_id, processed_frame, face_results), block=False)
                        except:
                            pass
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"❌ Error in face processing thread: {e}")
                    time.sleep(0.1)
        
        # Thread 3: License Plate Recognition Processing
        def plate_processing_thread():
            """Process frames for license plate recognition"""
            logger.info("🚗 Thread 3: License Plate Recognition started")
            
            while True:
                try:
                    frame_id, frame = self.plate_processing_queue.get(timeout=1.0)
                    
                    start_time = time.time()
                    plate_results = self._process_plates(frame)
                    processing_time = time.time() - start_time
                    
                    if plate_results:
                        self.stats['plate_detections'] += len(plate_results)
                        logger.debug(f"🚗 Detected {len(plate_results)} plates in {processing_time:.3f}s")
                    
                    try:
                        self.plate_results_queue.put((frame_id, plate_results), block=False)
                    except queue.Full:
                        try:
                            self.plate_results_queue.get(block=False)
                            self.plate_results_queue.put((frame_id, plate_results), block=False)
                        except:
                            pass
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"❌ Error in plate processing thread: {e}")
                    time.sleep(0.1)
        
        # Thread 4: Results Merging and Frame Encoding - IMPROVED VERSION
        def merging_thread():
            """Merge results from face and plate processing"""
            logger.info("🔄 Thread 4: Results Merging started (IMPROVED)")
            
            pending_results = {
                'frames': {},
                'face': {},
                'plate': {}
            }
            
            while True:
                try:
                    mode = self.current_mode
                    
                    # Collect raw frames
                    try:
                        while True:
                            frame_id, frame = self.raw_frame_queue.get(block=False)
                            pending_results['frames'][frame_id] = frame
                    except queue.Empty:
                        pass
                    
                    # Collect face results
                    if mode in ["face", "both"]:
                        try:
                            while True:
                                frame_id, processed_frame, face_results = self.face_results_queue.get(block=False)
                                pending_results['face'][frame_id] = (processed_frame, face_results)
                        except queue.Empty:
                            pass
                    
                    # Collect plate results
                    if mode in ["plate", "both"]:
                        try:
                            while True:
                                frame_id, plate_results = self.plate_results_queue.get(block=False)
                                pending_results['plate'][frame_id] = plate_results
                        except queue.Empty:
                            pass
                    
                    # Process frames - IMPROVED LOGIC
                    frames_to_process = list(pending_results['frames'].keys())
                    
                    for frame_id in frames_to_process:
                        # FIX: More flexible matching - use partial results if available
                        # Increased timeout to 60 frames (~2 seconds at 30fps)
                        should_process = False
                        use_timeout = self._is_frame_too_old(frame_id, max_age=60)
                        
                        if mode == "face":
                            should_process = frame_id in pending_results['face'] or use_timeout
                        elif mode == "plate":
                            should_process = frame_id in pending_results['plate'] or use_timeout
                        elif mode == "both":
                            # FIX: Process if we have EITHER result, not just both
                            has_face = frame_id in pending_results['face']
                            has_plate = frame_id in pending_results['plate']
                            should_process = (has_face or has_plate) or use_timeout
                        
                        if should_process:
                            frame = pending_results['frames'].pop(frame_id)
                            
                            face_results = []
                            plate_results = []
                            final_frame = frame.copy()
                            
                            # Get face results if available
                            if frame_id in pending_results['face']:
                                processed_frame, face_results = pending_results['face'].pop(frame_id)
                                # Always use processed frame if available (has face labels)
                                final_frame = processed_frame
                            
                            # Get plate results and draw if available
                            if frame_id in pending_results['plate']:
                                plate_results = pending_results['plate'].pop(frame_id)
                                if plate_results:
                                    # Draw plates on top of face-annotated frame
                                    final_frame = self.plate_system.draw_outputs(final_frame, plate_results)
                            
                            # Encode and queue for sending
                            self._encode_and_queue_frame(final_frame, face_results, plate_results)
                            self.stats['frames_processed'] += 1
                    
                    # Clean up old cached results
                    current_frame_id = self.frame_id
                    self._cleanup_old_results(pending_results, current_frame_id, max_age=90)
                    
                    time.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"❌ Error in merging thread: {e}")
                    time.sleep(0.1)
        
        # Start all threads
        threads = [
            threading.Thread(target=capture_thread, daemon=True, name="CaptureThread"),
            threading.Thread(target=face_processing_thread, daemon=True, name="FaceProcessingThread"),
            threading.Thread(target=plate_processing_thread, daemon=True, name="PlateProcessingThread"),
            threading.Thread(target=merging_thread, daemon=True, name="MergingThread")
        ]
        
        for thread in threads:
            thread.start()
            logger.info(f"✅ Started {thread.name}")
    
    def _process_faces(self, frame) -> Tuple[np.ndarray, List[Dict]]:
        """Process faces in frame"""
        try:
            return self.face_system.process_frame(frame)
        except Exception as e:
            logger.error(f"Face processing error: {e}")
            return frame, []
    
    def _process_plates(self, frame) -> List[Dict]:
        """Process license plates in frame"""
        try:
            return self.plate_system.process_frame(frame)
        except Exception as e:
            logger.error(f"Plate processing error: {e}")
            return []
    
    def _is_frame_too_old(self, frame_id: int, max_age: int = 60) -> bool:
        """Check if frame is too old to wait for results"""
        current_frame_id = self.frame_id
        return (current_frame_id - frame_id) > max_age
    
    def _cleanup_old_results(self, pending_results: Dict, current_frame_id: int, max_age: int = 90):
        """Remove old cached results"""
        for result_type in ['frames', 'face', 'plate']:
            old_frame_ids = [fid for fid in pending_results[result_type].keys() 
                           if (current_frame_id - fid) > max_age]
            for fid in old_frame_ids:
                pending_results[result_type].pop(fid, None)
    
    def _encode_and_queue_frame(self, frame, face_results, plate_results):
        """Encode frame to JPEG and add to output queue"""
        try:
            height, width = frame.shape[:2]
            if width > 800:
                scale = 800 / width
                new_width = 800
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            message = {
                "type": "video_frame",
                "frame": frame_base64,
                "face_results": convert_to_serializable(face_results),
                "plate_results": convert_to_serializable(plate_results),
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                self.processed_frame_queue.put(message, block=False)
            except queue.Full:
                try:
                    self.processed_frame_queue.get(block=False)
                    self.processed_frame_queue.put(message, block=False)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error encoding frame: {e}")
    
    def _clear_all_queues(self):
        """Clear all queues when stopping camera"""
        queues = [
            self.raw_frame_queue,
            self.face_processing_queue,
            self.plate_processing_queue,
            self.face_results_queue,
            self.plate_results_queue,
            self.processed_frame_queue
        ]
        
        for q in queues:
            while not q.empty():
                try:
                    q.get(block=False)
                except queue.Empty:
                    break


# Create the API instance
security_api = OptimizedSecuritySystemAPI()
app = security_api.app

if __name__ == "__main__":
    import uvicorn
    logger.info("🔒 Starting Fixed Security System API v2.0.1...")
    logger.info("📡 API: http://localhost:8000")
    logger.info("🔌 WebSocket: ws://localhost:8000/ws")
    logger.info("📚 Docs: http://localhost:8000/docs")
    logger.info("🧵 Multi-threaded processing: 4 threads")
    logger.info("   - Thread 1: Video Capture (FIXED)")
    logger.info("   - Thread 2: Facial Recognition")
    logger.info("   - Thread 3: License Plate Recognition")
    logger.info("   - Thread 4: Results Merging (IMPROVED)")
    
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