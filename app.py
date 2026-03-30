#!/usr/bin/env python3
"""
Enhanced FastAPI Backend for Security System with Demographic Analysis
and Human Action Recognition (HAR).

Key features:
1. Multi-threaded processing pipeline (7 threads)
2. Demographic analysis for unknown faces using DeepFace
3. Human Action Recognition using SlowFast R50
4. GPU-accelerated processing
5. Improved queue management and result merging
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
from har_system import HumanActionRecognitionSystem
from weapon_detection_system import WeaponDetectionSystem

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
class LoginRequest(BaseModel):
    username: str
    password: str

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

# Simple user credentials for dashboard login
DASHBOARD_USERS = {
    "admin": "admin123",
}

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
            self.is_running = True
            self.analysis_thread = threading.Thread(
                target=self._analysis_worker,
                daemon=True,
                name="DemographicsThread"
            )
            self.analysis_thread.start()
            logger.info("✅ Demographics analysis thread started")
    
    def _analysis_worker(self):
        """Background thread for demographics analysis"""
        while self.is_running:
            try:
                request = self.analysis_queue.get(timeout=1.0)
                request_id = request['request_id']
                face_image = request['face_image']
                
                demographics = self._analyze_face(face_image)
                
                if demographics:
                    with self.pending_lock:
                        self.pending_demographics[request_id] = demographics
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Error in demographics worker: {e}")
    
    def request_analysis(self, face_image: np.ndarray, request_id: int):
        """Queue a face image for demographics analysis"""
        if not self.enabled:
            return
        
        try:
            self.analysis_queue.put({
                'request_id': request_id,
                'face_image': face_image.copy()
            }, block=False)
        except queue.Full:
            pass
    
    def get_pending_result(self, request_id: int) -> Optional[Dict]:
        """Get demographics result for a request_id if available"""
        with self.pending_lock:
            return self.pending_demographics.pop(request_id, None)
    
    def _analyze_face(self, face_image: np.ndarray) -> Dict:
        """Analyze a face image for demographics"""
        try:
            if face_image is None or face_image.size == 0:
                return {}
            
            # Check cache
            cache_key = hash(face_image.tobytes()[:1000])
            current_time = time.time()
            
            if cache_key in self.last_analysis_cache:
                cached_time, cached_result = self.last_analysis_cache[cache_key]
                if current_time - cached_time < self.cache_duration:
                    return cached_result
            
            # Run DeepFace analysis
            result = DeepFace.analyze(
                face_image,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(result, list):
                result = result[0]
            
            demographics = {
                'age': result.get('age', 'unknown'),
                'gender': result.get('dominant_gender', 'unknown'),
                'emotion': result.get('dominant_emotion', 'unknown'),
                'gender_confidence': result.get('gender', {}).get(result.get('dominant_gender', ''), 0),
                'emotion_confidence': result.get('emotion', {}).get(result.get('dominant_emotion', ''), 0)
            }

            self.last_analysis_cache[cache_key] = (current_time, demographics)
            self._clean_cache(current_time)
            
            return demographics
            
        except Exception as e:
            logger.error(f"❌ Error in face demographics analysis: {e}")
            return {}
    
    def _clean_cache(self, current_time):
        """Remove old cache entries"""
        keys_to_remove = []
        for key, (timestamp, _) in self.last_analysis_cache.items():
            if current_time - timestamp > self.cache_duration * 2:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.last_analysis_cache[key]
    
    def stop(self):
        """Stop analysis thread"""
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
        self.app = FastAPI(title="Security System API", version="4.0.0")
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
        
        # Initialize Human Action Recognition system
        har_model_path = "har/best_model.pth"
        try:
            import os
            abs_path = os.path.abspath(har_model_path)
            logger.info(f"🏃 Looking for HAR model at: {abs_path}")
            self.har_system = HumanActionRecognitionSystem(
                model_path=har_model_path,
                device="auto",
                confidence_threshold=0.5,
                clip_interval_frames=30,
            )
            self.har_enabled = True
            logger.info("✅ Human Action Recognition system initialized")
        except FileNotFoundError as e:
            logger.warning(f"⚠️ HAR model not found: {e}")
            logger.warning(f"   Expected at: {os.path.abspath(har_model_path)}")
            logger.warning(f"   Current working directory: {os.getcwd()}")
            self.har_system = None
            self.har_enabled = False
        except Exception as e:
            logger.warning(f"⚠️ HAR system not available: {type(e).__name__}: {e}")
            self.har_system = None
            self.har_enabled = False
        
        # Initialize Weapon Detection system
        weapon_model_path = "weapon_detection_model/best.pt"
        try:
            self.weapon_system = WeaponDetectionSystem(model_path=weapon_model_path)
            self.weapon_detection_enabled = True
            logger.info("✅ Weapon detection system initialized")
        except FileNotFoundError as e:
            logger.warning(f"⚠️ Weapon detection model not found: {e}")
            self.weapon_system = None
            self.weapon_detection_enabled = False
        except Exception as e:
            logger.warning(f"⚠️ Weapon detection system not available: {type(e).__name__}: {e}")
            self.weapon_system = None
            self.weapon_detection_enabled = False

        # Initialize demographics analyzer
        self.demographics_analyzer = FaceDemographicsAnalyzer()
        self.demographics_enabled = True
        
        # Initialize face detector for finding all faces
        self.face_detector = YuNetFaceDetector()
        
         # Video streaming - Multi-camera support
        self.video_capture = None          # kept for backward compatibility (CAM-01 / laptop)
        self.is_streaming = False
        
        # Multi-camera management
        # key = camera_id (e.g. "CAM-01", "CAM-02"), value = dict with capture, url, active
        self.cameras = {}                  # managed IP / extra cameras
        self.cameras_lock = threading.Lock()
        
        # Independent detection flags - allows any combination
        self.face_detection_enabled = True
        self.plate_detection_enabled = True
        
        # Multi-threaded queue system
        self.raw_frame_queue = queue.Queue(maxsize=10)
        self.face_processing_queue = queue.Queue(maxsize=10)
        self.plate_processing_queue = queue.Queue(maxsize=10)
        self.fire_processing_queue = queue.Queue(maxsize=10)
        self.har_processing_queue = queue.Queue(maxsize=10)      # NEW: HAR queue
        self.weapon_processing_queue = queue.Queue(maxsize=10)   # Weapon queue
        self.face_results_queue = queue.Queue(maxsize=10)
        self.plate_results_queue = queue.Queue(maxsize=10)
        self.fire_results_queue = queue.Queue(maxsize=10)
        self.har_results_queue = queue.Queue(maxsize=10)          # NEW: HAR results queue
        self.weapon_results_queue = queue.Queue(maxsize=10)       # Weapon results queue
        self.processed_frame_queue = queue.Queue(maxsize=10)
        
        self.client_connections: List[WebSocket] = []
        
        # Frame management
        self.frame_skip = 1
        self.frame_counter = 0
        self.frame_id = 0
        self.demographics_request_id = 0
        
        # Track unknown faces across frames
        self.unknown_faces_cache = {}
        self.unknown_faces_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'face_detections': 0,
            'plate_detections': 0,
            'fire_detections': 0,
            'har_detections': 0,          # NEW
            'weapon_detections': 0,
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

        @self.app.post("/api/login")
        async def login(credentials: LoginRequest):
            """Authenticate user for dashboard access"""
            if (credentials.username in DASHBOARD_USERS and
                    DASHBOARD_USERS[credentials.username] == credentials.password):
                logger.info(f"✅ User '{credentials.username}' logged in")
                return {
                    "status": "success",
                    "user": {
                        "username": credentials.username,
                        "role": "administrator"
                    }
                }
            raise HTTPException(status_code=401, detail="Invalid username or password")

        @self.app.get("/")
        async def root():
            return {
                "message": "Security System API v4.0.0 (with Demographics + HAR)", 
                "status": "online",
                "features": {
                    "face_recognition": True,
                    "license_plate_recognition": True,
                    "demographics_analysis": DEEPFACE_AVAILABLE,
                    "fire_detection": self.fire_system is not None,
                    "human_action_recognition": self.har_system is not None,
                    "weapon_detection": self.weapon_system is not None
                }
            }
        
        @self.app.get("/api/status")
        async def get_status():
            stats = self.db.get_statistics()
            # Check IP camera (CAM-02) status
            cam02_streaming = False
            with self.cameras_lock:
                if "CAM-02" in self.cameras:
                    cam_info = self.cameras["CAM-02"]
                    cam02_streaming = cam_info.get("active", False) and cam_info.get("capture") is not None
            return {
                "status": "online",
                "streaming": self.is_streaming,
                "cam01_streaming": self.is_streaming and self.video_capture is not None and self.video_capture.isOpened(),
                "cam02_streaming": cam02_streaming,
                "face_detection_enabled": self.face_detection_enabled,
                "plate_detection_enabled": self.plate_detection_enabled,
                "demographics_enabled": self.demographics_enabled,
                "fire_detection_enabled": self.fire_detection_enabled,
                "fire_system_available": self.fire_system is not None,
                "har_enabled": self.har_enabled,
                "har_system_available": self.har_system is not None,
                "weapon_detection_enabled": self.weapon_detection_enabled,
                "weapon_system_available": self.weapon_system is not None,
                "statistics": convert_to_serializable(stats),
                "performance": convert_to_serializable(self.stats)
            }
        
        @self.app.post("/api/camera/start")
        async def start_camera():
            try:
                if not self.is_streaming:
                    self.video_capture = cv2.VideoCapture(0)
                    if self.video_capture.isOpened():
                        self.is_streaming = True
                        logger.info("📹 Camera started")
                        return {"status": "success", "message": "Camera started"}
                    else:
                        raise HTTPException(status_code=500, detail="Failed to open camera")
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
                        self.video_capture = None
                    self._clear_all_queues()
                    logger.info("🛑 Laptop camera stopped")
                    return {"status": "success", "message": "Laptop camera stopped"}
                else:
                    return {"status": "info", "message": "Laptop camera not running"}
            except Exception as e:
                logger.error(f"❌ Error stopping camera: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/camera/stop-all")
        async def stop_all_cameras():
            """Stop all cameras (laptop + IP)"""
            try:
                self.is_streaming = False
                if self.video_capture:
                    self.video_capture.release()
                    self.video_capture = None
                with self.cameras_lock:
                    for cam_id, cam_info in self.cameras.items():
                        if cam_info.get("capture"):
                            cam_info["capture"].release()
                            cam_info["active"] = False
                    self.cameras.clear()
                self._clear_all_queues()
                logger.info("🛑 All cameras stopped")
                return {"status": "success", "message": "All cameras stopped"}
            except Exception as e:
                logger.error(f"❌ Error stopping all cameras: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/camera/add-ip")
        async def add_ip_camera(request_data: dict):
            """Add an IP camera (e.g. phone via IP Webcam app)"""
            try:
                url = request_data.get("url")          # e.g. "http://192.168.1.5:8080/video"
                camera_id = request_data.get("camera_id", "CAM-02")
                location = request_data.get("location", "IP Camera")
                
                if not url:
                    raise HTTPException(status_code=400, detail="URL is required")
                
                # Try to open the stream
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    raise HTTPException(status_code=500, detail=f"Could not open video stream: {url}")
                
                with self.cameras_lock:
                    # If camera_id already exists, release the old one
                    if camera_id in self.cameras:
                        old = self.cameras[camera_id]
                        if old.get("capture"):
                            old["capture"].release()
                    
                    self.cameras[camera_id] = {
                        "capture": cap,
                        "url": url,
                        "location": location,
                        "active": True,
                    }
                
                logger.info(f"📱 IP Camera added: {camera_id} -> {url}")
                return {
                    "status": "success",
                    "message": f"IP Camera {camera_id} connected",
                    "camera_id": camera_id,
                    "url": url,
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"❌ Error adding IP camera: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/camera/remove-ip")
        async def remove_ip_camera(request_data: dict):
            """Remove / disconnect an IP camera"""
            try:
                camera_id = request_data.get("camera_id", "CAM-02")
                
                with self.cameras_lock:
                    if camera_id in self.cameras:
                        cam = self.cameras.pop(camera_id)
                        if cam.get("capture"):
                            cam["capture"].release()
                        logger.info(f"📱 IP Camera removed: {camera_id}")
                        return {"status": "success", "message": f"Camera {camera_id} removed"}
                    else:
                        return {"status": "info", "message": f"Camera {camera_id} not found"}
            except Exception as e:
                logger.error(f"❌ Error removing IP camera: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/cameras/list")
        async def list_cameras():
            """List all active cameras"""
            result = []
            
            # Laptop camera (CAM-01)
            result.append({
                "camera_id": "CAM-01",
                "type": "local",
                "location": "Main Entrance",
                "active": self.is_streaming and self.video_capture is not None and self.video_capture.isOpened(),
            })
            
            # IP cameras
            with self.cameras_lock:
                for cam_id, cam_info in self.cameras.items():
                    result.append({
                        "camera_id": cam_id,
                        "type": "ip",
                        "url": cam_info.get("url", ""),
                        "location": cam_info.get("location", ""),
                        "active": cam_info.get("active", False) and cam_info.get("capture") is not None,
                    })
            
            return {"cameras": result}

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
            logger.info(f"🚗 License plate detection: {'enabled' if enabled else 'disabled'}")
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
        
        # ── NEW: HAR toggle endpoint ──────────────────────────────────
        @self.app.post("/api/har/toggle")
        async def toggle_har(request: dict):
            """Toggle Human Action Recognition on/off"""
            enabled = request.get("enabled", True)
            self.har_enabled = enabled
            logger.info(f"🏃 Human Action Recognition: {'enabled' if enabled else 'disabled'}")
            if self.har_system is None:
                logger.warning("⚠️ HAR toggled but model is not loaded. "
                               "Place your trained model at har/best_model.pth")
                return {
                    "status": "warning",
                    "har_enabled": self.har_enabled,
                    "message": "HAR toggled but model is not loaded. "
                               "Place your trained model at har/best_model.pth and restart the server."
                }
            return {"status": "success", "har_enabled": self.har_enabled}
        
        # ── NEW: HAR statistics endpoint ──────────────────────────────
        @self.app.get("/api/har/stats")
        async def get_har_stats():
            """Get HAR system statistics"""
            if self.har_system is None:
                return {"available": False, "message": "HAR system not loaded"}
            return {
                "available": True,
                "enabled": self.har_enabled,
                "statistics": convert_to_serializable(self.har_system.get_statistics())
            }
        
        # ── Weapon Detection toggle endpoint ─────────────────────
        @self.app.post("/api/weapon/toggle")
        async def toggle_weapon_detection(request: dict):
            """Toggle Weapon Detection on/off"""
            enabled = request.get("enabled", True)
            self.weapon_detection_enabled = enabled
            logger.info(f"🔫 Weapon Detection: {'enabled' if enabled else 'disabled'}")
            if self.weapon_system is None:
                logger.warning("⚠️ Weapon detection toggled but model is not loaded.")
                return {
                    "status": "warning",
                    "weapon_detection_enabled": self.weapon_detection_enabled,
                    "message": "Weapon detection toggled but model is not loaded. "
                               "Place your trained model at weapon_detection_model/best.pt and restart the server."
                }
            return {"status": "success", "weapon_detection_enabled": self.weapon_detection_enabled}

        # ── Weapon Detection statistics endpoint ──────────────────
        @self.app.get("/api/weapon/stats")
        async def get_weapon_stats():
            """Get Weapon Detection system statistics"""
            if self.weapon_system is None:
                return {"available": False, "message": "Weapon detection system not loaded"}
            return {
                "available": True,
                "enabled": self.weapon_detection_enabled,
                "statistics": convert_to_serializable(self.weapon_system.get_statistics())
            }

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
        """Setup multi-threaded processing pipeline with demographics and HAR"""
        
        # ── Thread 1: Video Capture ───────────────────────────────────
        def capture_thread():
            """Capture frames from ALL cameras and queue for processing"""
            logger.info("🎥 Thread 1: Video Capture started (multi-camera)")
            
            while True:
                try:
                    captured_any = False
                    
                    # ── Camera 1: Laptop webcam (CAM-01) ──────────────
                    if self.is_streaming and self.video_capture and self.video_capture.isOpened():
                        ret, frame = self.video_capture.read()
                        if ret:
                            captured_any = True
                            self.stats['frames_captured'] += 1
                            current_frame_id = self.frame_id
                            self.frame_id += 1
                            
                            self.frame_counter += 1
                            if self.frame_counter % self.frame_skip == 0:
                                self._dispatch_frame(current_frame_id, "CAM-01", frame)
                        else:
                            pass  # failed read, will retry
                    
                    # ── IP Cameras (CAM-02, CAM-03, …) ────────────────
                    with self.cameras_lock:
                        cam_items = list(self.cameras.items())
                    
                    for cam_id, cam_info in cam_items:
                        if not cam_info.get("active"):
                            continue
                        cap = cam_info.get("capture")
                        if cap is None or not cap.isOpened():
                            # Try to reconnect
                            url = cam_info.get("url")
                            if url:
                                try:
                                    new_cap = cv2.VideoCapture(url)
                                    if new_cap.isOpened():
                                        with self.cameras_lock:
                                            self.cameras[cam_id]["capture"] = new_cap
                                        cap = new_cap
                                        logger.info(f"📱 Reconnected {cam_id}")
                                    else:
                                        continue
                                except:
                                    continue
                            else:
                                continue
                        
                        ret, frame = cap.read()
                        if ret:
                            captured_any = True
                            self.stats['frames_captured'] += 1
                            current_frame_id = self.frame_id
                            self.frame_id += 1
                            self._dispatch_frame(current_frame_id, cam_id, frame)
                        else:
                            logger.warning(f"⚠️ Failed to read frame from {cam_id}")
                    
                    if not captured_any:
                        time.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"❌ Error in capture thread: {e}")
                    time.sleep(0.1)
        
        # ── Thread 2: Facial Recognition + Demographics ───────────────
        def face_processing_thread():
            """Process frames for facial recognition and demographics"""
            logger.info("👤 Thread 2: Face Recognition + Demographics started")
            
            while True:
                try:
                    frame_id, camera_id, frame = self.face_processing_queue.get(timeout=1.0)
                    
                    start_time = time.time()
                    
                    # Process faces
                    processed_frame, face_results = self._process_faces(frame)
                    
                    # Demographics for unknown faces
                    if self.demographics_enabled and face_results:
                        for result in face_results:
                            if result.get('name') == 'Unknown':
                                self.stats['unknown_faces'] += 1
                                bbox = result.get('bbox', result.get('bounding_box', None))
                                if bbox:
                                    x, y, w, h = bbox
                                    face_crop = frame[max(0,y):y+h, max(0,x):x+w]
                                    if face_crop.size > 0:
                                        self.demographics_request_id += 1
                                        req_id = self.demographics_request_id
                                        self.demographics_analyzer.request_analysis(face_crop, req_id)
                                        
                                        # Try to get cached demographics
                                        bbox_key = self._get_bbox_key(bbox)
                                        cached, _ = self._find_matching_unknown_face(bbox)
                                        if cached and cached.get('demographics'):
                                            result.update(cached['demographics'])
                                            self.stats['demographics_analyzed'] += 1
                    
                    processing_time = time.time() - start_time
                    
                    if face_results:
                        self.stats['face_detections'] += len(face_results)
                        logger.debug(f"👤 Detected {len(face_results)} faces in {processing_time:.3f}s")
                    
                    try:
                        self.face_results_queue.put((frame_id, camera_id, processed_frame, face_results), block=False)
                    except queue.Full:
                        try:
                            self.face_results_queue.get(block=False)
                            self.face_results_queue.put((frame_id, camera_id, processed_frame, face_results), block=False)
                        except:
                            pass
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"❌ Error in face processing thread: {e}")
                    time.sleep(0.1)
        
        # ── Thread 3: License Plate Recognition ───────────────────────
        def plate_processing_thread():
            """Process frames for license plate recognition"""
            logger.info("🚗 Thread 3: License Plate Recognition started")
            
            while True:
                try:
                    frame_id, camera_id, frame = self.plate_processing_queue.get(timeout=1.0)
                    
                    start_time = time.time()
                    plate_results = self._process_plates(frame)
                    processing_time = time.time() - start_time
                    
                    if plate_results:
                        self.stats['plate_detections'] += len(plate_results)
                        logger.debug(f"🚗 Detected {len(plate_results)} plates in {processing_time:.3f}s")
                    
                    try:
                        self.plate_results_queue.put((frame_id, camera_id, plate_results), block=False)
                    except queue.Full:
                        try:
                            self.plate_results_queue.get(block=False)
                            self.plate_results_queue.put((frame_id, camera_id, plate_results), block=False)
                        except:
                            pass
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"❌ Error in plate processing thread: {e}")
                    time.sleep(0.1)
        
        # ── Thread 4: Fire Detection Processing ───────────────────────
        def fire_processing_thread():
            """Process frames for fire and smoke detection"""
            logger.info("🔥 Thread 4: Fire Detection started")
            
            while True:
                try:
                    frame_id, camera_id, frame = self.fire_processing_queue.get(timeout=1.0)
                    
                    if not self.fire_detection_enabled or not self.fire_system:
                        continue
                    
                    start_time = time.time()
                    fire_results = self.fire_system.process_frame(frame)
                    processing_time = time.time() - start_time
                    
                    if fire_results:
                        self.stats['fire_detections'] += len(fire_results)
                        logger.debug(f"🔥 Detected {len(fire_results)} fire/smoke in {processing_time:.3f}s")
                        
                        critical = [r for r in fire_results if r.get('severity') == 'critical']
                        if critical:
                            logger.warning(f"🚨 CRITICAL FIRE ALERT! {len(critical)} critical detection(s)")
                    
                    try:
                        self.fire_results_queue.put((frame_id, camera_id, fire_results), block=False)
                    except queue.Full:
                        try:
                            self.fire_results_queue.get(block=False)
                            self.fire_results_queue.put((frame_id, camera_id, fire_results), block=False)
                        except:
                            pass
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"❌ Error in fire processing thread: {e}")
                    time.sleep(0.1)
        
        # ── Thread 5: Human Action Recognition Processing ─────────────
        def har_processing_thread():
            """Process frames for human action recognition (SlowFast)"""
            logger.info("🏃 Thread 5: Human Action Recognition started")
            
            while True:
                try:
                    frame_id, camera_id, frame = self.har_processing_queue.get(timeout=1.0)
                    
                    if not self.har_enabled or not self.har_system:
                        continue
                    
                    start_time = time.time()
                    har_results = self.har_system.process_frame(frame)
                    processing_time = time.time() - start_time
                    
                    if har_results:
                        # Only count non-normal detections for stats
                        non_normal = [r for r in har_results if r.get('class') != 'normal']
                        if non_normal:
                            self.stats['har_detections'] += len(non_normal)
                            logger.debug(f"🏃 HAR: {non_normal[0]['class']} "
                                         f"({non_normal[0]['confidence']:.2f}) in {processing_time:.3f}s")
                        
                        critical = [r for r in har_results if r.get('severity') == 'critical']
                        if critical:
                            logger.warning(f"🚨 CRITICAL HAR ALERT! "
                                           f"{critical[0]['action_label']} detected!")
                    
                    try:
                        self.har_results_queue.put((frame_id, camera_id, har_results), block=False)
                    except queue.Full:
                        try:
                            self.har_results_queue.get(block=False)
                            self.har_results_queue.put((frame_id, camera_id, har_results), block=False)
                        except:
                            pass
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"❌ Error in HAR processing thread: {e}")
                    time.sleep(0.1)
        
        # ── Thread 6: Weapon Detection Processing ────────────────────
        def weapon_processing_thread():
            """Process frames for weapon detection"""
            logger.info("🔫 Thread 6: Weapon Detection started")

            while True:
                try:
                    frame_id, camera_id, frame = self.weapon_processing_queue.get(timeout=1.0)

                    if not self.weapon_detection_enabled or not self.weapon_system:
                        continue

                    start_time = time.time()
                    weapon_results = self.weapon_system.process_frame(frame)
                    processing_time = time.time() - start_time

                    if weapon_results:
                        self.stats['weapon_detections'] += len(weapon_results)
                        logger.debug(f"🔫 Detected {len(weapon_results)} weapon(s) in {processing_time:.3f}s")

                        critical = [r for r in weapon_results if r.get('severity') == 'critical']
                        if critical:
                            logger.warning(f"🚨 CRITICAL WEAPON ALERT! {len(critical)} critical detection(s)")

                    try:
                        self.weapon_results_queue.put((frame_id, camera_id, weapon_results), block=False)
                    except queue.Full:
                        try:
                            self.weapon_results_queue.get(block=False)
                            self.weapon_results_queue.put((frame_id, camera_id, weapon_results), block=False)
                        except:
                            pass

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"❌ Error in weapon processing thread: {e}")
                    time.sleep(0.1)

        # ── Thread 7: Results Merging and Frame Encoding ──────────────
        def merging_thread():
            """Merge results from face, plate, fire, and HAR processing"""
            logger.info("🔄 Thread 6: Results Merging started")
            
            pending_results = {
                'frames': {},
                'face': {},
                'plate': {},
                'fire': {},
                'har': {},   # NEW
                'weapon': {},
            }
            
            while True:
                try:
                    # Collect raw frames
                    try:
                        while True:
                            frame_id, camera_id, frame = self.raw_frame_queue.get(block=False)
                            pending_results['frames'][frame_id] = (camera_id, frame)
                    except queue.Empty:
                        pass
                    
                    # Collect face results
                    if self.face_detection_enabled:
                        try:
                            while True:
                                frame_id, camera_id, processed_frame, face_results = self.face_results_queue.get(block=False)
                                pending_results['face'][frame_id] = (processed_frame, face_results)
                        except queue.Empty:
                            pass
                    
                    # Collect plate results
                    if self.plate_detection_enabled:
                        try:
                            while True:
                                frame_id, camera_id, plate_results = self.plate_results_queue.get(block=False)
                                pending_results['plate'][frame_id] = plate_results
                        except queue.Empty:
                            pass
                    
                    # Collect fire results
                    if self.fire_detection_enabled and self.fire_system:
                        try:
                            while True:
                                frame_id, camera_id, fire_results = self.fire_results_queue.get(block=False)
                                pending_results['fire'][frame_id] = fire_results
                        except queue.Empty:
                            pass
                    
                    # NEW: Collect HAR results
                    if self.har_enabled and self.har_system:
                        try:
                            while True:
                                frame_id, camera_id, har_results = self.har_results_queue.get(block=False)
                                pending_results['har'][frame_id] = har_results
                        except queue.Empty:
                            pass

                    # Collect weapon results
                    if self.weapon_detection_enabled and self.weapon_system:
                        try:
                            while True:
                                frame_id, camera_id, weapon_results = self.weapon_results_queue.get(block=False)
                                pending_results['weapon'][frame_id] = weapon_results
                        except queue.Empty:
                            pass

                    # Process frames
                    frames_to_process = list(pending_results['frames'].keys())
                    
                    for frame_id in frames_to_process:
                        should_process = False
                        use_timeout = self._is_frame_too_old(frame_id, max_age=60)
                        
                        expected_results_ready = True
                        
                        if self.face_detection_enabled:
                            expected_results_ready = expected_results_ready and (
                                frame_id in pending_results['face'] or use_timeout)
                        
                        if self.plate_detection_enabled:
                            expected_results_ready = expected_results_ready and (
                                frame_id in pending_results['plate'] or use_timeout)
                        
                        # Fire and HAR results are optional (merged when available)
                        
                        if not self.face_detection_enabled and not self.plate_detection_enabled:
                            should_process = True
                        else:
                            should_process = expected_results_ready
                        
                        if should_process:
                            camera_id, frame = pending_results['frames'].pop(frame_id)
                            
                            face_results = []
                            plate_results = []
                            fire_results = []
                            har_results = []       # NEW
                            weapon_results = []
                            final_frame = frame.copy()
                            
                            # Get face results
                            if frame_id in pending_results['face']:
                                processed_frame, face_results = pending_results['face'].pop(frame_id)
                                final_frame = processed_frame
                            
                            # Get plate results and draw
                            if frame_id in pending_results['plate']:
                                plate_results = pending_results['plate'].pop(frame_id)
                                if plate_results:
                                    final_frame = self.plate_system.draw_outputs(final_frame, plate_results)
                            
                            # Get fire results and draw
                            if frame_id in pending_results['fire']:
                                fire_results = pending_results['fire'].pop(frame_id)
                                if fire_results:
                                    final_frame = self.fire_system.draw_detections(final_frame, fire_results)
                            
                            # NEW: Get HAR results and draw
                            if frame_id in pending_results['har']:
                                har_results = pending_results['har'].pop(frame_id)
                                if har_results:
                                    final_frame = self.har_system.draw_detections(final_frame, har_results)

                            # Get weapon results and draw
                            if frame_id in pending_results['weapon']:
                                weapon_results = pending_results['weapon'].pop(frame_id)
                                if weapon_results:
                                    final_frame = self.weapon_system.draw_detections(final_frame, weapon_results)

                            # Encode and queue for sending
                            self._encode_and_queue_frame(
                                final_frame, face_results, plate_results,
                                fire_results, har_results, weapon_results,
                                camera_id=camera_id
                            )
                            self.stats['frames_processed'] += 1
                    
                    # Clean up old results
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
            threading.Thread(target=har_processing_thread, daemon=True, name="HARProcessingThread"),     # NEW
            threading.Thread(target=weapon_processing_thread, daemon=True, name="WeaponProcessingThread"),
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
        
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        overlap_area = x_overlap * y_overlap
        
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - overlap_area
        
        if union_area == 0:
            return False
        
        iou = overlap_area / union_area
        return iou > threshold
    
    def _get_bbox_key(self, bbox, grid_size=50):
        """Generate a key for bbox to track same face across frames"""
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        grid_x = center_x // grid_size
        grid_y = center_y // grid_size
        return f"{grid_x}_{grid_y}_{w//10}_{h//10}"
    
    def _find_matching_unknown_face(self, bbox):
        """Find if this bbox matches a previously detected unknown face"""
        with self.unknown_faces_lock:
            bbox_key = self._get_bbox_key(bbox)
            
            if bbox_key in self.unknown_faces_cache:
                entry = self.unknown_faces_cache[bbox_key]
                return entry, bbox_key
            
            # Check neighbours
            for key, entry in self.unknown_faces_cache.items():
                if entry.get('bbox') and self._bboxes_overlap(bbox, entry['bbox'], threshold=0.3):
                    return entry, key
        
        return None, None
    
    def _is_frame_too_old(self, frame_id: int, max_age: int = 60) -> bool:
        """Check if frame is too old to wait for results"""
        return (self.frame_id - frame_id) > max_age
    
    def _cleanup_old_results(self, pending_results: Dict, current_frame_id: int, max_age: int = 90):
        """Remove old cached results"""
        for result_type in ['frames', 'face', 'plate', 'fire', 'har', 'weapon']:
            old_frame_ids = [fid for fid in pending_results[result_type].keys() 
                           if (current_frame_id - fid) > max_age]
            for fid in old_frame_ids:
                pending_results[result_type].pop(fid, None)
    
    def _dispatch_frame(self, frame_id, camera_id, frame):
        """Dispatch a single frame from any camera into the processing queues"""
        queued_for_face = False
        queued_for_plate = False
        queued_for_fire = False
        queued_for_har = False
        queued_for_weapon = False

        if self.face_detection_enabled:
            try:
                self.face_processing_queue.put((frame_id, camera_id, frame.copy()), block=False)
                queued_for_face = True
            except queue.Full:
                self.stats['queue_skips'] += 1
        else:
            queued_for_face = True

        if self.plate_detection_enabled:
            try:
                self.plate_processing_queue.put((frame_id, camera_id, frame.copy()), block=False)
                queued_for_plate = True
            except queue.Full:
                self.stats['queue_skips'] += 1
        else:
            queued_for_plate = True

        if self.fire_detection_enabled and self.fire_system:
            try:
                self.fire_processing_queue.put((frame_id, camera_id, frame.copy()), block=False)
                queued_for_fire = True
            except queue.Full:
                self.stats['queue_skips'] += 1
        else:
            queued_for_fire = True

        if self.har_enabled and self.har_system:
            try:
                self.har_processing_queue.put((frame_id, camera_id, frame.copy()), block=False)
                queued_for_har = True
            except queue.Full:
                self.stats['queue_skips'] += 1
        else:
            queued_for_har = True

        if self.weapon_detection_enabled and self.weapon_system:
            try:
                self.weapon_processing_queue.put((frame_id, camera_id, frame.copy()), block=False)
                queued_for_weapon = True
            except queue.Full:
                self.stats['queue_skips'] += 1
        else:
            queued_for_weapon = True

        if queued_for_face and queued_for_plate and queued_for_fire and queued_for_har and queued_for_weapon:
            try:
                self.raw_frame_queue.put((frame_id, camera_id, frame.copy()), block=False)
            except queue.Full:
                self.stats['frames_dropped'] += 1
        else:
            self.stats['frames_dropped'] += 1
    
    def _encode_and_queue_frame(self, frame, face_results, plate_results,
                                 fire_results=None, har_results=None, weapon_results=None,
                                 camera_id="CAM-01"):
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
                "camera_id": camera_id,
                "frame": frame_base64,
                "face_results": convert_to_serializable(face_results),
                "plate_results": convert_to_serializable(plate_results),
                "fire_results": convert_to_serializable(fire_results) if fire_results else [],
                "har_results": convert_to_serializable(har_results) if har_results else [],   # NEW
                "weapon_results": convert_to_serializable(weapon_results) if weapon_results else [],
                "timestamp": datetime.now().isoformat(),
                "demographics_enabled": self.demographics_enabled,
                "fire_detection_enabled": self.fire_detection_enabled,
                "har_enabled": self.har_enabled,                                                # NEW
                "weapon_detection_enabled": self.weapon_detection_enabled,
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
            self.har_processing_queue,       # NEW
            self.weapon_processing_queue,
            self.face_results_queue,
            self.plate_results_queue,
            self.fire_results_queue,
            self.har_results_queue,           # NEW
            self.weapon_results_queue,
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
    logger.info("🔒 Starting Enhanced Security System API v4.0.0...")
    logger.info("📡 API: http://localhost:8000")
    logger.info("🔌 WebSocket: ws://localhost:8000/ws")
    logger.info("📚 Docs: http://localhost:8000/docs")
    logger.info("🧵 Multi-threaded processing: 8 threads")
    logger.info("   - Thread 1: Video Capture")
    logger.info("   - Thread 2: Facial Recognition + Demographics")
    logger.info("   - Thread 3: License Plate Recognition")
    logger.info("   - Thread 4: Fire & Smoke Detection")
    logger.info("   - Thread 5: Human Action Recognition (SlowFast)")
    logger.info("   - Thread 6: Weapon Detection")
    logger.info("   - Thread 7: Results Merging")
    logger.info("   - Thread 8: Demographics Analysis (DeepFace)")
    
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