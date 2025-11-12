#!/usr/bin/env python3
"""
Enhanced FastAPI Backend for Security System with Demographic Analysis
Key features:
1. Multi-threaded processing pipeline
2. Demographic analysis for unknown faces using DeepFace
3. GPU-accelerated processing
4. Improved queue management and result merging
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
from face_detection import YuNetFaceDetector
from fire_detection_system import FireDetectionSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import DeepFace for demographics analysis
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("✅ DeepFace available for demographic analysis")
except ImportError:
    DEEPFACE_AVAILABLE = False
    logger.warning("⚠️ DeepFace not available - demographic analysis disabled")

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
    'database': 'facial_recognition',
    'user': 'postgres',
    'password': 'incorect'
}


class FaceDemographicsAnalyzer:
    """
    Class to handle face demographics analysis using DeepFace
    GPU-accelerated when available
    """
    
    def __init__(self):
        self.enabled = DEEPFACE_AVAILABLE
        self.last_analysis_cache = {}
        self.cache_duration = 3.0  # Cache results for 3 seconds
        self.analysis_queue = queue.Queue(maxsize=20)
        self.results_queue = queue.Queue(maxsize=20)
        self.pending_demographics = {}  # Store demographics results until they're used
        self.pending_lock = threading.Lock()  # Thread-safe access to pending results
        self.is_running = False
        
        if self.enabled:
            # Start analysis thread
            self.analysis_thread = threading.Thread(
                target=self._analysis_worker, 
                daemon=True, 
                name="DemographicsAnalysisThread"
            )
            self.is_running = True
            self.analysis_thread.start()
            logger.info("🧠 Demographics analysis thread started")

    def _analysis_worker(self):
        """Worker thread for demographic analysis"""
        while self.is_running:
            try:
                # Get analysis request from queue
                request_id, face_image, face_bbox = self.analysis_queue.get(timeout=1.0)
                
                # Perform analysis
                demographics = self._analyze_face_internal(face_image, face_bbox)
                
                # Put result in results queue
                try:
                    self.results_queue.put((request_id, demographics), block=False)
                except queue.Full:
                    # Drop oldest result if queue is full
                    try:
                        self.results_queue.get(block=False)
                        self.results_queue.put((request_id, demographics), block=False)
                    except:
                        pass
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Error in demographics worker: {e}")
                time.sleep(0.1)

    def _analyze_face_internal(self, face_image: np.ndarray, face_bbox: tuple) -> dict:
        """Internal method to analyze face"""
        try:
            x, y, w, h = face_bbox
            cache_key = f"{x}_{y}_{w}_{h}"
            current_time = time.time()
            
            # Check cache
            if cache_key in self.last_analysis_cache:
                last_time, last_result = self.last_analysis_cache[cache_key]
                if current_time - last_time < self.cache_duration:
                    return last_result
            
            # Prepare image
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_image
                
            # Ensure minimum size for DeepFace
            height, width = face_rgb.shape[:2]
            if width < 48 or height < 48:
                scale = max(48/width, 48/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                face_rgb = cv2.resize(face_rgb, (new_width, new_height))

            # Analyze with DeepFace
            analysis = DeepFace.analyze(
                img_path=face_rgb,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False,
                silent=True
            )
            
            # Extract results
            if isinstance(analysis, list) and len(analysis) > 0:
                result = analysis[0]
            else:
                result = analysis
                
            demographics = {
                'age': int(result.get('age', 0)),
                'gender': result.get('dominant_gender', 'unknown'),
                'emotion': result.get('dominant_emotion', 'unknown'),
                'gender_confidence': result.get('gender', {}).get(result.get('dominant_gender', ''), 0),
                'emotion_confidence': result.get('emotion', {}).get(result.get('dominant_emotion', ''), 0)
            }

            # Cache result
            self.last_analysis_cache[cache_key] = (current_time, demographics)
            self._clean_cache(current_time)
            
            return demographics
            
        except Exception as e:
            logger.debug(f"Demographics analysis error: {e}")
            return {}
    
    def analyze_face_async(self, face_image: np.ndarray, face_bbox: tuple, request_id: int) -> None:
        """
        Queue face for async demographic analysis
        Non-blocking method for thread-safe analysis
        """
        if not self.enabled:
            return
        
        try:
            self.analysis_queue.put((request_id, face_image.copy(), face_bbox), block=False)
        except queue.Full:
            logger.debug("Demographics analysis queue full, skipping")
    
    def get_results(self, timeout: float = 0.01) -> Dict[int, dict]:
        """
        Get available analysis results and store them in pending_demographics
        Returns: Dict of newly arrived results
        """
        new_results = {}
        if not self.enabled:
            return new_results
        
        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                request_id, demographics = self.results_queue.get(block=False)
                # Store in pending results with timestamp
                with self.pending_lock:
                    self.pending_demographics[request_id] = {
                        'demographics': demographics,
                        'timestamp': time.time()
                    }
                new_results[request_id] = demographics
            except queue.Empty:
                break
        
        return new_results
    
    def get_pending_result(self, request_id: int) -> Optional[dict]:
        """
        Get a pending demographics result by request_id
        Returns the demographics dict if found, None otherwise
        Does NOT remove the result from pending (allows multiple checks)
        """
        with self.pending_lock:
            if request_id in self.pending_demographics:
                return self.pending_demographics[request_id]['demographics']
        return None
    
    def consume_pending_result(self, request_id: int) -> Optional[dict]:
        """
        Get and remove a pending demographics result by request_id
        Use this when you've successfully matched and used a result
        """
        with self.pending_lock:
            if request_id in self.pending_demographics:
                result = self.pending_demographics[request_id]['demographics']
                del self.pending_demographics[request_id]
                return result
        return None
    
    def cleanup_old_pending_results(self, max_age: float = 10.0):
        """Remove pending results older than max_age seconds"""
        current_time = time.time()
        with self.pending_lock:
            expired_ids = [
                req_id for req_id, data in self.pending_demographics.items()
                if current_time - data['timestamp'] > max_age
            ]
            for req_id in expired_ids:
                del self.pending_demographics[req_id]
                logger.debug(f"Cleaned up expired demographics result for request_id={req_id}")
    
    def _clean_cache(self, current_time):
        """Remove old cache entries"""
        keys_to_remove = []
        for key, (timestamp, _) in self.last_analysis_cache.items():
            if current_time - timestamp > self.cache_duration * 2:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.last_analysis_cache[key]
    
    def stop(self):
        """Stop the analysis thread"""
        self.is_running = False
        if hasattr(self, 'analysis_thread'):
            self.analysis_thread.join(timeout=2.0)


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


class EnhancedSecuritySystemAPI:
    def __init__(self):
        self.app = FastAPI(title="Security System API", version="3.0.0")
        self.setup_cors()
        self.setup_routes()
        
        # Initialize systems
        self.db = DatabaseManager(**DB_CONFIG)
        self.db.connect()
        
        self.face_system = FacialRecognitionSystem(DB_CONFIG, camera_id="0")
        self.plate_system = LicensePlateRecognitionSystem(DB_CONFIG)
        
        # Initialize fire detection system
        try:
            self.fire_system = FireDetectionSystem()
            self.fire_detection_enabled = True
            logger.info("✅ Fire detection system initialized")
        except Exception as e:
            logger.warning(f"⚠️ Fire detection system not available: {e}")
            self.fire_system = None
            self.fire_detection_enabled = False
        
        # Initialize demographics analyzer
        self.demographics_analyzer = FaceDemographicsAnalyzer()
        self.demographics_enabled = True
        
        # Initialize face detector for finding all faces
        self.face_detector = YuNetFaceDetector()
        
        # Video streaming
        self.video_capture = None
        self.is_streaming = False
        
        # Independent detection flags - allows any combination
        self.face_detection_enabled = True
        self.plate_detection_enabled = True
        
        # Multi-threaded queue system
        self.raw_frame_queue = queue.Queue(maxsize=10)
        self.face_processing_queue = queue.Queue(maxsize=10)
        self.plate_processing_queue = queue.Queue(maxsize=10)
        self.fire_processing_queue = queue.Queue(maxsize=10)
        self.face_results_queue = queue.Queue(maxsize=10)
        self.plate_results_queue = queue.Queue(maxsize=10)
        self.fire_results_queue = queue.Queue(maxsize=10)
        self.processed_frame_queue = queue.Queue(maxsize=10)
        
        self.client_connections: List[WebSocket] = []
        
        # Frame management
        self.frame_skip = 1
        self.frame_counter = 0
        self.frame_id = 0
        self.demographics_request_id = 0
        
        # Track unknown faces across frames
        self.unknown_faces_cache = {}  # key: bbox hash, value: {request_id, last_seen, demographics}
        self.unknown_faces_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'face_detections': 0,
            'plate_detections': 0,
            'fire_detections': 0,
            'frames_dropped': 0,
            'queue_skips': 0,
            'demographics_analyzed': 0,
            'unknown_faces': 0
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
            return {
                "message": "Security System API v3.0.0 (with Demographics)", 
                "status": "online",
                "features": {
                    "face_recognition": True,
                    "license_plate_recognition": True,
                    "demographics_analysis": DEEPFACE_AVAILABLE,
                    "fire_detection": self.fire_system is not None
                }
            }
        
        @self.app.get("/api/status")
        async def get_status():
            stats = self.db.get_statistics()
            return {
                "status": "online",
                "streaming": self.is_streaming,
                "face_detection_enabled": self.face_detection_enabled,
                "plate_detection_enabled": self.plate_detection_enabled,
                "demographics_enabled": self.demographics_enabled and DEEPFACE_AVAILABLE,
                "fire_detection_enabled": self.fire_detection_enabled,
                "fire_system_available": self.fire_system is not None,
                "statistics": stats,
                "connected_clients": len(self.client_connections),
                "performance": {
                    "frames_captured": self.stats['frames_captured'],
                    "frames_processed": self.stats['frames_processed'],
                    "face_detections": self.stats['face_detections'],
                    "plate_detections": self.stats['plate_detections'],
                    "fire_detections": self.stats['fire_detections'],
                    "frames_dropped": self.stats['frames_dropped'],
                    "queue_skips": self.stats['queue_skips'],
                    "demographics_analyzed": self.stats['demographics_analyzed'],
                    "unknown_faces": self.stats['unknown_faces']
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
        
        @self.app.post("/api/face/toggle")
        async def toggle_face_detection(request: dict):
            """Toggle face detection on/off"""
            enabled = request.get("enabled", True)
            self.face_detection_enabled = enabled
            logger.info(f"👤 Face detection: {'enabled' if enabled else 'disabled'}")
            return {"status": "success", "face_detection_enabled": self.face_detection_enabled}
        
        @self.app.post("/api/plate/toggle")
        async def toggle_plate_detection(request: dict):
            """Toggle license plate detection on/off"""
            enabled = request.get("enabled", True)
            self.plate_detection_enabled = enabled
            logger.info(f"� License plate detection: {'enabled' if enabled else 'disabled'}")
            return {"status": "success", "plate_detection_enabled": self.plate_detection_enabled}
        
        @self.app.post("/api/demographics/toggle")
        async def toggle_demographics(request: dict):
            """Toggle demographic analysis on/off"""
            enabled = request.get("enabled", True)
            self.demographics_enabled = enabled
            logger.info(f"🧠 Demographics analysis: {'enabled' if enabled else 'disabled'}")
            return {"status": "success", "demographics_enabled": self.demographics_enabled}
        
        @self.app.post("/api/fire/toggle")
        async def toggle_fire_detection(request: dict):
            """Toggle fire detection on/off"""
            if self.fire_system is None:
                raise HTTPException(status_code=400, detail="Fire detection system not available")
            enabled = request.get("enabled", True)
            self.fire_detection_enabled = enabled
            logger.info(f"🔥 Fire detection: {'enabled' if enabled else 'disabled'}")
            return {"status": "success", "fire_detection_enabled": self.fire_detection_enabled}
        
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
        """Setup multi-threaded processing pipeline with demographics"""
        
        # Thread 1: Video Capture
        def capture_thread():
            """Capture frames and queue for processing"""
            logger.info("🎥 Thread 1: Video Capture started")
            
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
                            
                            # Track if frame was successfully queued for processing
                            queued_for_face = False
                            queued_for_plate = False
                            queued_for_fire = False
                            
                            # Try to queue for face processing
                            if self.face_detection_enabled:
                                try:
                                    self.face_processing_queue.put((current_frame_id, frame.copy()), block=False)
                                    queued_for_face = True
                                except queue.Full:
                                    self.stats['queue_skips'] += 1
                            else:
                                queued_for_face = True
                            
                            # Try to queue for plate processing
                            if self.plate_detection_enabled:
                                try:
                                    self.plate_processing_queue.put((current_frame_id, frame.copy()), block=False)
                                    queued_for_plate = True
                                except queue.Full:
                                    self.stats['queue_skips'] += 1
                            else:
                                queued_for_plate = True
                            
                            # Try to queue for fire detection (always if enabled)
                            if self.fire_detection_enabled and self.fire_system:
                                try:
                                    self.fire_processing_queue.put((current_frame_id, frame.copy()), block=False)
                                    queued_for_fire = True
                                except queue.Full:
                                    self.stats['queue_skips'] += 1
                            else:
                                queued_for_fire = True
                            
                            # Always add frame to raw queue if successfully queued for all enabled processing
                            # If no detection is enabled, frames still go through
                            if queued_for_face and queued_for_plate and queued_for_fire:
                                try:
                                    self.raw_frame_queue.put((current_frame_id, frame.copy()), block=False)
                                except queue.Full:
                                    self.stats['frames_dropped'] += 1
                            else:
                                self.stats['frames_dropped'] += 1
                        else:
                            logger.warning("⚠️ Failed to read frame from camera")
                            time.sleep(0.1)
                    else:
                        time.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"❌ Error in capture thread: {e}")
                    time.sleep(0.1)
        
        # Thread 2: Enhanced Facial Recognition Processing with Demographics
        def face_processing_thread():
            """Process frames for facial recognition and demographics"""
            logger.info("👤 Thread 2: Face Recognition + Demographics started")
            
            while True:
                try:
                    frame_id, frame = self.face_processing_queue.get(timeout=1.0)
                    
                    start_time = time.time()
                    
                    # Process recognized faces
                    processed_frame, face_results = self._process_faces(frame)
                    
                    # If demographics enabled, analyze unknown faces
                    if self.demographics_enabled and self.demographics_analyzer.enabled:
                        # Detect ALL faces in the frame
                        all_faces = self.face_detector.detect_faces(frame)
                        
                        # Get bounding boxes of recognized faces
                        recognized_bboxes = [result['bbox'] for result in face_results if 'bbox' in result]
                        
                        logger.debug(f"🔍 Found {len(all_faces)} total faces, {len(recognized_bboxes)} recognized")
                        
                        # Find unknown faces
                        unknown_faces = []
                        for face_bbox in all_faces:
                            is_recognized = False
                            
                            # Check if this face overlaps with any recognized face
                            for recognized_bbox in recognized_bboxes:
                                if self._bboxes_overlap(face_bbox, recognized_bbox):
                                    is_recognized = True
                                    break
                            
                            if not is_recognized:
                                unknown_faces.append(face_bbox)
                        
                        if unknown_faces:
                            logger.info(f"👤 Detected {len(unknown_faces)} unknown face(s)")
                        
                        # Queue unknown faces for demographic analysis
                        for face_bbox in unknown_faces:
                            x, y, w, h = face_bbox
                            face_region = frame[y:y+h, x:x+w]
                            
                            if face_region.size > 0:
                                # Check if we've seen this face before
                                cached_entry, cache_key = self._find_matching_unknown_face(face_bbox)
                                
                                if cached_entry:
                                    # Reuse existing request_id and demographics
                                    request_id = cached_entry['request_id']
                                    demographics = cached_entry.get('demographics', {})
                                    
                                    # Update last seen time
                                    with self.unknown_faces_lock:
                                        self.unknown_faces_cache[cache_key]['last_seen'] = time.time()
                                        self.unknown_faces_cache[cache_key]['bbox'] = face_bbox
                                    
                                    logger.debug(f"🔄 Reusing request_id={request_id} for tracked unknown face")
                                else:
                                    # New unknown face - generate unique request ID
                                    request_id = self.demographics_request_id
                                    self.demographics_request_id += 1
                                    demographics = {}
                                    
                                    # Queue for async analysis
                                    self.demographics_analyzer.analyze_face_async(
                                        face_region, face_bbox, request_id
                                    )
                                    
                                    # Add to cache
                                    bbox_key = self._get_bbox_key(face_bbox)
                                    with self.unknown_faces_lock:
                                        self.unknown_faces_cache[bbox_key] = {
                                            'request_id': request_id,
                                            'bbox': face_bbox,
                                            'last_seen': time.time(),
                                            'demographics': {}
                                        }
                                    
                                    logger.debug(f"🆕 New unknown face with request_id={request_id}")
                                    self.stats['unknown_faces'] += 1
                                
                                # Add result for unknown face
                                unknown_result = {
                                    'name': 'Unknown',
                                    'employee_id': '',
                                    'department': '',
                                    'confidence': 0.0,
                                    'bbox': face_bbox,
                                    'timestamp': datetime.now(),
                                    'demographics_request_id': request_id,
                                    'age': demographics.get('age', ''),
                                    'gender': demographics.get('gender', ''),
                                    'emotion': demographics.get('emotion', '')
                                }
                                face_results.append(unknown_result)
                        
                        # Check for newly completed demographic analyses
                        # This collects results from the queue and stores them in pending_demographics
                        demographics_results = self.demographics_analyzer.get_results(timeout=0.01)
                        
                        # Update face results with demographics (check both new and pending results)
                        for result in face_results:
                            if 'demographics_request_id' in result:
                                req_id = result['demographics_request_id']
                                
                                # First check newly arrived results
                                demographics = None
                                if req_id in demographics_results:
                                    demographics = demographics_results[req_id]
                                else:
                                    # Check pending results from previous frames
                                    demographics = self.demographics_analyzer.get_pending_result(req_id)
                                
                                if demographics:
                                    result.update({
                                        'age': demographics.get('age', ''),
                                        'gender': demographics.get('gender', ''),
                                        'emotion': demographics.get('emotion', '')
                                    })
                                    
                                    logger.info(f"✅ Applied demographics for request_id={req_id}: Age={demographics.get('age')}, Gender={demographics.get('gender')}, Emotion={demographics.get('emotion')}")
                                    
                                    # Update the cache so future frames can reuse this data
                                    bbox = result.get('bbox')
                                    if bbox:
                                        bbox_key = self._get_bbox_key(bbox)
                                        with self.unknown_faces_lock:
                                            if bbox_key in self.unknown_faces_cache:
                                                self.unknown_faces_cache[bbox_key]['demographics'] = demographics
                                    
                                    self.stats['demographics_analyzed'] += 1
                        
                        # Periodically clean up old cached unknown faces and pending results
                        if self.frame_id % 100 == 0:
                            self.demographics_analyzer.cleanup_old_pending_results(max_age=10.0)
                            self._cleanup_old_unknown_faces(max_age=5.0)
                        
                        # Draw demographics on frame for unknown faces
                        processed_frame = self._draw_demographics_on_frame(processed_frame, face_results)
                    
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
        
        # Thread 4: Fire Detection Processing
        def fire_processing_thread():
            """Process frames for fire and smoke detection"""
            logger.info("🔥 Thread 4: Fire Detection started")
            
            while True:
                try:
                    frame_id, frame = self.fire_processing_queue.get(timeout=1.0)
                    
                    if not self.fire_detection_enabled or not self.fire_system:
                        continue
                    
                    start_time = time.time()
                    fire_results = self.fire_system.process_frame(frame)
                    processing_time = time.time() - start_time
                    
                    if fire_results:
                        self.stats['fire_detections'] += len(fire_results)
                        logger.debug(f"🔥 Detected {len(fire_results)} fire/smoke in {processing_time:.3f}s")
                        
                        # Log critical alerts
                        critical = [r for r in fire_results if r.get('severity') == 'critical']
                        if critical:
                            logger.warning(f"🚨 CRITICAL FIRE ALERT! {len(critical)} critical detection(s)")
                    
                    try:
                        self.fire_results_queue.put((frame_id, fire_results), block=False)
                    except queue.Full:
                        try:
                            self.fire_results_queue.get(block=False)
                            self.fire_results_queue.put((frame_id, fire_results), block=False)
                        except:
                            pass
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"❌ Error in fire processing thread: {e}")
                    time.sleep(0.1)
        
        # Thread 5: Results Merging and Frame Encoding
        def merging_thread():
            """Merge results from face, plate, and fire processing"""
            logger.info("🔄 Thread 5: Results Merging started")
            
            pending_results = {
                'frames': {},
                'face': {},
                'plate': {},
                'fire': {}
            }
            
            while True:
                try:
                    # Collect raw frames
                    try:
                        while True:
                            frame_id, frame = self.raw_frame_queue.get(block=False)
                            pending_results['frames'][frame_id] = frame
                    except queue.Empty:
                        pass
                    
                    # Collect face results if face detection is enabled
                    if self.face_detection_enabled:
                        try:
                            while True:
                                frame_id, processed_frame, face_results = self.face_results_queue.get(block=False)
                                pending_results['face'][frame_id] = (processed_frame, face_results)
                        except queue.Empty:
                            pass
                    
                    # Collect plate results if plate detection is enabled
                    if self.plate_detection_enabled:
                        try:
                            while True:
                                frame_id, plate_results = self.plate_results_queue.get(block=False)
                                pending_results['plate'][frame_id] = plate_results
                        except queue.Empty:
                            pass
                    
                    # Collect fire detection results if fire detection is enabled
                    if self.fire_detection_enabled and self.fire_system:
                        try:
                            while True:
                                frame_id, fire_results = self.fire_results_queue.get(block=False)
                                pending_results['fire'][frame_id] = fire_results
                        except queue.Empty:
                            pass
                    
                    # Process frames
                    frames_to_process = list(pending_results['frames'].keys())
                    
                    for frame_id in frames_to_process:
                        should_process = False
                        use_timeout = self._is_frame_too_old(frame_id, max_age=60)
                        
                        # Determine if we should process this frame
                        # Check if all ENABLED detections have results or timeout occurred
                        expected_results_ready = True
                        
                        if self.face_detection_enabled:
                            expected_results_ready = expected_results_ready and (frame_id in pending_results['face'] or use_timeout)
                        
                        if self.plate_detection_enabled:
                            expected_results_ready = expected_results_ready and (frame_id in pending_results['plate'] or use_timeout)
                        
                        # Fire detection results are optional (merged when available)
                        # If no detection is enabled, process immediately
                        if not self.face_detection_enabled and not self.plate_detection_enabled:
                            should_process = True
                        else:
                            should_process = expected_results_ready
                        
                        if should_process:
                            frame = pending_results['frames'].pop(frame_id)
                            
                            face_results = []
                            plate_results = []
                            fire_results = []
                            final_frame = frame.copy()
                            
                            # Get face results if available
                            if frame_id in pending_results['face']:
                                processed_frame, face_results = pending_results['face'].pop(frame_id)
                                final_frame = processed_frame
                            
                            # Get plate results and draw if available
                            if frame_id in pending_results['plate']:
                                plate_results = pending_results['plate'].pop(frame_id)
                                if plate_results:
                                    final_frame = self.plate_system.draw_outputs(final_frame, plate_results)
                            
                            # Get fire results and draw if available
                            if frame_id in pending_results['fire']:
                                fire_results = pending_results['fire'].pop(frame_id)
                                if fire_results:
                                    final_frame = self.fire_system.draw_detections(final_frame, fire_results)
                            
                            # Encode and queue for sending
                            self._encode_and_queue_frame(final_frame, face_results, plate_results, fire_results)
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
            threading.Thread(target=fire_processing_thread, daemon=True, name="FireProcessingThread"),
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
    
    def _bboxes_overlap(self, bbox1, bbox2, threshold=0.5):
        """Check if two bounding boxes overlap significantly"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate overlap area
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        overlap_area = x_overlap * y_overlap
        
        # Calculate areas
        area1 = w1 * h1
        area2 = w2 * h2
        
        # Calculate IoU
        union_area = area1 + area2 - overlap_area
        
        if union_area == 0:
            return False
        
        iou = overlap_area / union_area
        return iou > threshold
    
    def _get_bbox_key(self, bbox, grid_size=50):
        """
        Generate a key for bbox to track same face across frames
        Uses grid-based hashing for approximate location matching
        """
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Grid-based key to allow some movement
        grid_x = center_x // grid_size
        grid_y = center_y // grid_size
        
        return f"{grid_x}_{grid_y}_{w//10}_{h//10}"
    
    def _find_matching_unknown_face(self, bbox):
        """
        Find if this bbox matches a previously detected unknown face
        Returns (matched_cache_entry, cache_key) or (None, None)
        """
        with self.unknown_faces_lock:
            bbox_key = self._get_bbox_key(bbox)
            
            # Check exact match first
            if bbox_key in self.unknown_faces_cache:
                cache_entry = self.unknown_faces_cache[bbox_key]
                # Only reuse if seen recently (within 2 seconds)
                if time.time() - cache_entry['last_seen'] < 2.0:
                    return cache_entry, bbox_key
            
            # Check nearby grid cells for similar faces
            x, y, w, h = bbox
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    test_bbox = (x + dx*50, y + dy*50, w, h)
                    test_key = self._get_bbox_key(test_bbox)
                    
                    if test_key in self.unknown_faces_cache:
                        cache_entry = self.unknown_faces_cache[test_key]
                        if time.time() - cache_entry['last_seen'] < 2.0:
                            # Verify they actually overlap
                            if self._bboxes_overlap(bbox, cache_entry['bbox'], threshold=0.3):
                                return cache_entry, test_key
            
            return None, None
    
    def _draw_demographics_on_frame(self, frame, face_results):
        """Draw demographics information on frame for unknown faces"""
        output = frame.copy()
        
        for result in face_results:
            # Only draw demographics for unknown faces
            if result.get('name') == 'Unknown':
                bbox = result.get('bbox')
                if bbox:
                    x, y, w, h = bbox
                    
                    # Draw RED bounding box for unknown faces
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    
                    # Draw "Unknown" label at the top
                    label_text = "Unknown"
                    (label_width, label_height), _ = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    
                    # Draw background for "Unknown" label
                    cv2.rectangle(
                        output,
                        (x, y - label_height - 4),
                        (x + label_width, y),
                        (0, 0, 255),
                        -1
                    )
                    
                    # Draw "Unknown" text
                    cv2.putText(
                        output,
                        label_text,
                        (x, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
                    
                    # Get demographics - check if analysis is complete
                    age = result.get('age', '')
                    gender = result.get('gender', '')
                    emotion = result.get('emotion', '')
                    
                    # Prepare labels
                    labels = []
                    
                    # Always show demographic info or "Analyzing..." status
                    if age and gender and emotion:
                        # All demographics available
                        labels.append(f"Age: {age}")
                        labels.append(f"Gender: {gender}")
                        labels.append(f"Emotion: {emotion}")
                    elif age or gender or emotion:
                        # Partial demographics available
                        if age:
                            labels.append(f"Age: {age}")
                        if gender:
                            labels.append(f"Gender: {gender}")
                        if emotion:
                            labels.append(f"Emotion: {emotion}")
                    else:
                        # No demographics yet - show analyzing status
                        labels.append("Analyzing...")
                    
                    if labels:
                        # Draw background rectangle for text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        font_thickness = 1
                        padding = 5
                        line_height = 20
                        
                        # Calculate background size
                        max_width = 0
                        for label in labels:
                            (text_width, text_height), _ = cv2.getTextSize(
                                label, font, font_scale, font_thickness
                            )
                            max_width = max(max_width, text_width)
                        
                        bg_height = len(labels) * line_height + padding * 2
                        bg_width = max_width + padding * 2
                        
                        # Position below the face box
                        text_x = x
                        text_y = y + h + 5
                        
                        # Ensure text stays within frame bounds
                        if text_y + bg_height > output.shape[0]:
                            text_y = y - bg_height - 5
                        if text_x + bg_width > output.shape[1]:
                            text_x = output.shape[1] - bg_width - 5
                        
                        # Draw semi-transparent background
                        overlay = output.copy()
                        cv2.rectangle(
                            overlay,
                            (text_x, text_y),
                            (text_x + bg_width, text_y + bg_height),
                            (0, 0, 0),
                            -1
                        )
                        cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
                        
                        # Draw each label
                        for i, label in enumerate(labels):
                            y_offset = text_y + padding + (i + 1) * line_height - 5
                            # Use different color for "Analyzing..." 
                            color = (128, 128, 128) if label == "Analyzing..." else (0, 255, 255)  # Gray for analyzing, Yellow for results
                            cv2.putText(
                                output,
                                label,
                                (text_x + padding, y_offset),
                                font,
                                font_scale,
                                color,
                                font_thickness
                            )
        
        return output
    
    def _cleanup_old_unknown_faces(self, max_age: float = 5.0):
        """Remove old unknown face cache entries"""
        current_time = time.time()
        with self.unknown_faces_lock:
            keys_to_remove = [
                key for key, entry in self.unknown_faces_cache.items()
                if current_time - entry['last_seen'] > max_age
            ]
            for key in keys_to_remove:
                del self.unknown_faces_cache[key]
            
            if keys_to_remove:
                logger.debug(f"🧹 Cleaned up {len(keys_to_remove)} old unknown face cache entries")
    
    def _is_frame_too_old(self, frame_id: int, max_age: int = 60) -> bool:
        """Check if frame is too old to wait for results"""
        current_frame_id = self.frame_id
        return (current_frame_id - frame_id) > max_age
    
    def _cleanup_old_results(self, pending_results: Dict, current_frame_id: int, max_age: int = 90):
        """Remove old cached results"""
        for result_type in ['frames', 'face', 'plate', 'fire']:
            old_frame_ids = [fid for fid in pending_results[result_type].keys() 
                           if (current_frame_id - fid) > max_age]
            for fid in old_frame_ids:
                pending_results[result_type].pop(fid, None)
    
    def _encode_and_queue_frame(self, frame, face_results, plate_results, fire_results=None):
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
                "fire_results": convert_to_serializable(fire_results) if fire_results else [],
                "timestamp": datetime.now().isoformat(),
                "demographics_enabled": self.demographics_enabled,
                "fire_detection_enabled": self.fire_detection_enabled
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
            self.fire_processing_queue,
            self.face_results_queue,
            self.plate_results_queue,
            self.fire_results_queue,
            self.processed_frame_queue
        ]
        
        for q in queues:
            while not q.empty():
                try:
                    q.get(block=False)
                except queue.Empty:
                    break
    
    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, 'demographics_analyzer'):
            self.demographics_analyzer.stop()


# Create the API instance
security_api = EnhancedSecuritySystemAPI()
app = security_api.app

if __name__ == "__main__":
    import uvicorn
    logger.info("🔒 Starting Enhanced Security System API v3.0.0...")
    logger.info("📡 API: http://localhost:8000")
    logger.info("🔌 WebSocket: ws://localhost:8000/ws")
    logger.info("📚 Docs: http://localhost:8000/docs")
    logger.info("🧵 Multi-threaded processing: 6 threads")
    logger.info("   - Thread 1: Video Capture")
    logger.info("   - Thread 2: Facial Recognition + Demographics")
    logger.info("   - Thread 3: License Plate Recognition")
    logger.info("   - Thread 4: Fire & Smoke Detection")
    logger.info("   - Thread 5: Results Merging")
    logger.info("   - Thread 6: Demographics Analysis (DeepFace)")
    
    if DEEPFACE_AVAILABLE:
        logger.info("✅ DeepFace enabled - demographics analysis available")
    else:
        logger.info("⚠️ DeepFace not available - install with: pip install deepface")
    
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