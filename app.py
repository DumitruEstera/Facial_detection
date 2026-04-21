#!/usr/bin/env python3
"""
Enhanced FastAPI Backend for Security System with Demographic Analysis
and Human Action Recognition (HAR).

Key features:
1. Multi-threaded processing pipeline (7 threads)
2. Demographic analysis (age/gender/emotion) via InsightFace + FER
3. Human Action Recognition using SlowFast R50
4. GPU-accelerated processing
5. Improved queue management and result merging
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import csv
import io
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
import jwt
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import logging

from facial_recognition_system import FacialRecognitionSystem
from license_plate_recognition_system import LicensePlateRecognitionSystem
from database_manager import DatabaseManager
from fire_detection_system import FireDetectionSystem
from har_system import HumanActionRecognitionSystem
from weapon_detection_system import WeaponDetectionSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Demographic analysis (age/gender/emotion) is produced inline by
# FacialRecognitionSystem.process_frame via InsightFace + FER. No separate
# DeepFace pipeline is required.

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

class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = 'user'
    full_name: Optional[str] = None

class UpdatePersonRequest(BaseModel):
    name: Optional[str] = None
    department: Optional[str] = None
    authorized_zones: Optional[List[str]] = None

class UpdateLicensePlateRequest(BaseModel):
    owner_name: Optional[str] = None
    owner_id: Optional[str] = None
    vehicle_type: Optional[str] = None
    is_authorized: Optional[bool] = None
    expiry_date: Optional[str] = None
    notes: Optional[str] = None

class UpdateUserRequest(BaseModel):
    role: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[str] = None

class ZoneCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    is_restricted: bool = False

class ZoneUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_restricted: Optional[bool] = None

class CameraCreateRequest(BaseModel):
    camera_id: str
    name: Optional[str] = None
    zone_id: Optional[int] = None
    location: Optional[str] = None
    stream_url: Optional[str] = None
    camera_type: Optional[str] = 'ip'

class CameraUpdateRequest(BaseModel):
    name: Optional[str] = None
    zone_id: Optional[int] = None
    clear_zone: Optional[bool] = False
    location: Optional[str] = None
    stream_url: Optional[str] = None

class UpdateAlarmRequest(BaseModel):
    status: Optional[str] = None
    notes: Optional[str] = None

class BulkUpdateAlarmsRequest(BaseModel):
    alarm_ids: List[int]
    status: str

# JWT Configuration
JWT_SECRET = "security-dashboard-secret-key-change-in-production"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 8

security = HTTPBearer()

def create_jwt_token(user_data: dict) -> str:
    payload = {
        "sub": user_data["username"],
        "user_id": user_data["id"],
        "role": user_data["role"],
        "full_name": user_data.get("full_name", ""),
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_jwt_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    return decode_jwt_token(credentials.credentials)

def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'facial_recognition',
    'user': 'postgres',
    'password': 'incorect'
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

        # Demographics (age/gender/emotion) are produced by
        # FacialRecognitionSystem.process_frame; no separate analyzer is needed.
        self.demographics_enabled = True

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
        
        # Alarm deduplication: tracks last alarm time per (camera_id, type)
        self.alarm_cooldowns = {}
        self.alarm_cooldown_seconds = 30
        self.alarm_lock = threading.Lock()

        # Zone authorization caches (refreshed on change or every 30s)
        self._zone_cache_lock = threading.Lock()
        self._camera_zone_cache = {}      # camera_id -> {zone_id, zone_name, is_restricted} or None
        self._person_zones_cache = {}     # employee_id -> set(zone_name)
        self._zone_cache_refreshed_at = 0
        self._zone_cache_ttl = 30.0

        # Ensure CAM-01 (laptop) exists in camera registry
        try:
            self.db.upsert_camera(camera_id="CAM-01", name="Laptop Camera",
                                  location="Laptop Camera", camera_type='local')
        except Exception as e:
            logger.warning(f"Could not upsert CAM-01 into cameras registry: {e}")
        # Detection-log rate limiting (avoid one row per frame per detection)
        self.log_cooldowns = {}
        self.log_cooldown_seconds = 5
        self.log_lock = threading.Lock()

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
            user = self.db.authenticate_user(credentials.username, credentials.password)
            if user:
                token = create_jwt_token(user)
                logger.info(f"✅ User '{credentials.username}' logged in (role: {user['role']})")
                return {
                    "status": "success",
                    "token": token,
                    "user": {
                        "id": user["id"],
                        "username": user["username"],
                        "role": user["role"],
                        "full_name": user.get("full_name", "")
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
                    "demographics_analysis": True,
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

                # Persist into the cameras registry so zone assignment survives restart
                try:
                    self.db.upsert_camera(
                        camera_id=camera_id, name=location,
                        location=location, stream_url=url, camera_type='ip'
                    )
                    self._refresh_zone_caches(force=True)
                except Exception as _e:
                    logger.warning(f"Could not persist camera {camera_id}: {_e}")

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
                
                cap = None
                removed = False
                with self.cameras_lock:
                    if camera_id in self.cameras:
                        cam = self.cameras.pop(camera_id)
                        cap = cam.get("capture")
                        removed = True
                        logger.info(f"📱 IP Camera removed: {camera_id}")

                if removed:
                    # Release capture with a delay so the capture thread
                    # finishes any in-progress read on this object first.
                    if cap:
                        import threading as _threading
                        _threading.Timer(1.0, lambda c=cap: c.release()).start()
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
        async def toggle_face_detection(request: dict, current_user: dict = Depends(require_admin)):
            """Toggle face detection on/off (admin only)"""
            enabled = request.get("enabled", True)
            self.face_detection_enabled = enabled
            logger.info(f"👤 Face detection: {'enabled' if enabled else 'disabled'}")
            return {"status": "success", "face_detection_enabled": self.face_detection_enabled}
        
        @self.app.post("/api/plate/toggle")
        async def toggle_plate_detection(request: dict, current_user: dict = Depends(require_admin)):
            """Toggle license plate detection on/off (admin only)"""
            enabled = request.get("enabled", True)
            self.plate_detection_enabled = enabled
            logger.info(f"🚗 License plate detection: {'enabled' if enabled else 'disabled'}")
            return {"status": "success", "plate_detection_enabled": self.plate_detection_enabled}
        
        @self.app.post("/api/demographics/toggle")
        async def toggle_demographics(request: dict, current_user: dict = Depends(require_admin)):
            """Toggle demographic analysis on/off (admin only)"""
            enabled = request.get("enabled", True)
            self.demographics_enabled = enabled
            logger.info(f"🧠 Demographics analysis: {'enabled' if enabled else 'disabled'}")
            return {"status": "success", "demographics_enabled": self.demographics_enabled}
        
        @self.app.post("/api/fire/toggle")
        async def toggle_fire_detection(request: dict, current_user: dict = Depends(require_admin)):
            """Toggle fire detection on/off (admin only)"""
            if self.fire_system is None:
                raise HTTPException(status_code=400, detail="Fire detection system not available")
            enabled = request.get("enabled", True)
            self.fire_detection_enabled = enabled
            logger.info(f"🔥 Fire detection: {'enabled' if enabled else 'disabled'}")
            return {"status": "success", "fire_detection_enabled": self.fire_detection_enabled}
        
        # ── NEW: HAR toggle endpoint ──────────────────────────────────
        @self.app.post("/api/har/toggle")
        async def toggle_har(request: dict, current_user: dict = Depends(require_admin)):
            """Toggle Human Action Recognition on/off (admin only)"""
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
        async def toggle_weapon_detection(request: dict, current_user: dict = Depends(require_admin)):
            """Toggle Weapon Detection on/off (admin only)"""
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
        async def register_person(person: PersonRegistration, current_user: dict = Depends(require_admin)):
            try:
                person_id = self.db.add_person(
                    name=person.name,
                    employee_id=person.employee_id,
                    department=person.department,
                    authorized_zones=person.authorized_zones
                )
                self._refresh_zone_caches(force=True)
                return {
                    "status": "success",
                    "message": f"Person {person.name} registered successfully",
                    "person_id": person_id,
                    "person": person.dict()
                }
            except Exception as e:
                if "duplicate key" in str(e).lower() or "unique" in str(e).lower():
                    raise HTTPException(status_code=400, detail=f"Employee ID '{person.employee_id}' already exists")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/persons")
        async def list_persons(search: str = None, department: str = None,
                              limit: int = 50, offset: int = 0,
                              current_user: dict = Depends(require_admin)):
            try:
                result = self.db.list_persons(search=search, department=department,
                                             limit=limit, offset=offset)
                return convert_to_serializable(result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/persons/departments")
        async def get_departments(current_user: dict = Depends(require_admin)):
            try:
                departments = self.db.get_all_departments()
                return {"departments": departments}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/persons/{person_id}")
        async def get_person(person_id: int, current_user: dict = Depends(require_admin)):
            try:
                person = self.db.get_person_by_id(person_id)
                if not person:
                    raise HTTPException(status_code=404, detail="Person not found")
                face_count = self.db.count_person_embeddings(person_id)
                access_history = self.db.get_person_access_history(person_id, limit=20)
                return convert_to_serializable({
                    "person": person,
                    "face_count": face_count,
                    "access_history": access_history
                })
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put("/api/persons/{person_id}")
        async def update_person(person_id: int, data: UpdatePersonRequest,
                               current_user: dict = Depends(require_admin)):
            try:
                person = self.db.get_person_by_id(person_id)
                if not person:
                    raise HTTPException(status_code=404, detail="Person not found")
                updated = self.db.update_person(
                    person_id,
                    name=data.name,
                    department=data.department,
                    authorized_zones=data.authorized_zones
                )
                if updated:
                    self._refresh_zone_caches(force=True)
                    return {"status": "success", "message": "Person updated successfully"}
                return {"status": "info", "message": "No changes made"}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/api/persons/{person_id}")
        async def delete_person(person_id: int, current_user: dict = Depends(require_admin)):
            try:
                person = self.db.get_person_by_id(person_id)
                if not person:
                    raise HTTPException(status_code=404, detail="Person not found")
                deleted = self.db.delete_person(person_id)
                if deleted:
                    # Rebuild FAISS index after deletion
                    self.face_system._load_embeddings_from_db()
                    return {"status": "success", "message": f"Person '{person['name']}' deleted successfully"}
                raise HTTPException(status_code=500, detail="Failed to delete person")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/persons/{person_id}/faces")
        async def upload_face_images(person_id: int,
                                     files: List[UploadFile] = File(...),
                                     current_user: dict = Depends(require_admin)):
            """Upload face images for a registered person"""
            try:
                person = self.db.get_person_by_id(person_id)
                if not person:
                    raise HTTPException(status_code=404, detail="Person not found")

                added_count = 0
                errors = []

                for file in files:
                    try:
                        contents = await file.read()
                        nparr = np.frombuffer(contents, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        if img is None:
                            errors.append(f"{file.filename}: Could not decode image")
                            continue

                        embedding = self.face_system.extract_embedding_from_image(img)
                        if embedding is None:
                            errors.append(f"{file.filename}: Expected exactly 1 face in image")
                            continue

                        self.db.add_face_embedding(person_id, embedding)
                        added_count += 1

                    except Exception as img_error:
                        errors.append(f"{file.filename}: {str(img_error)}")

                # Rebuild FAISS index with new embeddings
                if added_count > 0:
                    self.face_system._load_embeddings_from_db()

                return {
                    "status": "success",
                    "added": added_count,
                    "errors": errors,
                    "message": f"Added {added_count} face embedding(s) for {person['name']}"
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/persons/{person_id}/access-history")
        async def get_person_access_history(person_id: int, limit: int = 20,
                                           current_user: dict = Depends(require_admin)):
            try:
                person = self.db.get_person_by_id(person_id)
                if not person:
                    raise HTTPException(status_code=404, detail="Person not found")
                history = self.db.get_person_access_history(person_id, limit=limit)
                return convert_to_serializable({"history": history})
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/plates/register")
        async def register_plate(plate: LicensePlateRegistration, current_user: dict = Depends(require_admin)):
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
        
        # ── License Plate Management endpoints (admin only) ─────
        @self.app.get("/api/plates")
        async def list_plates(search: str = None, vehicle_type: str = None,
                              is_authorized: bool = None,
                              limit: int = 50, offset: int = 0,
                              current_user: dict = Depends(require_admin)):
            try:
                result = self.db.list_license_plates(
                    search=search, vehicle_type=vehicle_type,
                    is_authorized=is_authorized, limit=limit, offset=offset
                )
                return convert_to_serializable(result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/plates/vehicle-types")
        async def get_vehicle_types(current_user: dict = Depends(require_admin)):
            try:
                types = self.db.get_all_vehicle_types()
                return {"vehicle_types": types}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/plates/{plate_number}")
        async def get_plate(plate_number: str, current_user: dict = Depends(require_admin)):
            try:
                plate = self.db.get_license_plate(plate_number)
                if not plate:
                    raise HTTPException(status_code=404, detail="License plate not found")
                access_history = self.db.get_plate_access_history(plate_number, limit=20)
                return convert_to_serializable({
                    "plate": plate,
                    "access_history": access_history
                })
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put("/api/plates/{plate_number}")
        async def update_plate(plate_number: str, data: UpdateLicensePlateRequest,
                               current_user: dict = Depends(require_admin)):
            try:
                plate = self.db.get_license_plate(plate_number)
                if not plate:
                    raise HTTPException(status_code=404, detail="License plate not found")
                updated = self.db.update_license_plate(
                    plate_number,
                    owner_name=data.owner_name,
                    owner_id=data.owner_id,
                    vehicle_type=data.vehicle_type,
                    is_authorized=data.is_authorized,
                    expiry_date=data.expiry_date,
                    notes=data.notes
                )
                if updated:
                    return {"status": "success", "message": f"Plate {plate_number} updated successfully"}
                return {"status": "info", "message": "No changes made"}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/api/plates/{plate_number}")
        async def delete_plate(plate_number: str, current_user: dict = Depends(require_admin)):
            try:
                plate = self.db.get_license_plate(plate_number)
                if not plate:
                    raise HTTPException(status_code=404, detail="License plate not found")
                deleted = self.db.delete_license_plate(plate_number)
                if deleted:
                    return {"status": "success", "message": f"Plate '{plate_number}' deleted successfully"}
                raise HTTPException(status_code=500, detail="Failed to delete plate")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/plates/{plate_number}/access-history")
        async def get_plate_history(plate_number: str, limit: int = 20,
                                    current_user: dict = Depends(require_admin)):
            try:
                plate = self.db.get_license_plate(plate_number)
                if not plate:
                    raise HTTPException(status_code=404, detail="License plate not found")
                history = self.db.get_plate_access_history(plate_number, limit=limit)
                return convert_to_serializable({"history": history})
            except HTTPException:
                raise
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
        async def get_statistics(current_user: dict = Depends(get_current_user)):
            try:
                stats = self.db.get_statistics()
                return convert_to_serializable(stats)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # ── User Management endpoints (admin only) ────────────────
        @self.app.get("/api/users")
        async def list_users(current_user: dict = Depends(require_admin)):
            """List all users"""
            try:
                users = self.db.list_users()
                return {"users": convert_to_serializable(users)}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/users")
        async def create_user(user_req: CreateUserRequest, current_user: dict = Depends(require_admin)):
            """Create a new user"""
            try:
                if user_req.role not in ('admin', 'user'):
                    raise HTTPException(status_code=400, detail="Role must be 'admin' or 'user'")
                user_id = self.db.create_user(
                    username=user_req.username,
                    password=user_req.password,
                    role=user_req.role,
                    full_name=user_req.full_name
                )
                return {"status": "success", "user_id": user_id, "message": f"User '{user_req.username}' created"}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put("/api/users/{user_id}")
        async def update_user(user_id: int, user_req: UpdateUserRequest, current_user: dict = Depends(require_admin)):
            """Update a user"""
            try:
                if user_req.role and user_req.role not in ('admin', 'user'):
                    raise HTTPException(status_code=400, detail="Role must be 'admin' or 'user'")
                updated = self.db.update_user(
                    user_id=user_id,
                    role=user_req.role,
                    full_name=user_req.full_name,
                    password=user_req.password
                )
                if updated:
                    return {"status": "success", "message": "User updated"}
                raise HTTPException(status_code=404, detail="User not found")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/api/users/{user_id}")
        async def delete_user(user_id: int, current_user: dict = Depends(require_admin)):
            """Delete a user"""
            try:
                if current_user.get("user_id") == user_id:
                    raise HTTPException(status_code=400, detail="Cannot delete your own account")
                deleted = self.db.delete_user(user_id)
                if deleted:
                    return {"status": "success", "message": "User deleted"}
                raise HTTPException(status_code=404, detail="User not found")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put("/api/users/me/password")
        async def change_own_password(request: dict, current_user: dict = Depends(get_current_user)):
            """Change own password (any authenticated user)"""
            try:
                new_password = request.get("new_password")
                if not new_password or len(new_password) < 4:
                    raise HTTPException(status_code=400, detail="Password must be at least 4 characters")
                self.db.update_user(user_id=current_user["user_id"], password=new_password)
                return {"status": "success", "message": "Password changed"}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # ── Zones management endpoints ─────────────────────────────

        @self.app.get("/api/zones")
        async def list_zones(current_user: dict = Depends(get_current_user)):
            """List all zones (available to any logged-in user for dropdowns)."""
            try:
                zones = self.db.list_zones()
                return {"zones": convert_to_serializable(zones)}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/zones")
        async def create_zone(req: ZoneCreateRequest,
                              current_user: dict = Depends(require_admin)):
            try:
                if not req.name.strip():
                    raise HTTPException(status_code=400, detail="Zone name is required")
                zone_id = self.db.create_zone(
                    name=req.name.strip(),
                    description=req.description,
                    is_restricted=req.is_restricted,
                )
                self._refresh_zone_caches(force=True)
                return {"status": "success", "zone_id": zone_id,
                        "message": f"Zone '{req.name}' created"}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put("/api/zones/{zone_id}")
        async def update_zone(zone_id: int, req: ZoneUpdateRequest,
                              current_user: dict = Depends(require_admin)):
            try:
                updated = self.db.update_zone(
                    zone_id,
                    name=req.name.strip() if req.name else None,
                    description=req.description,
                    is_restricted=req.is_restricted,
                )
                self._refresh_zone_caches(force=True)
                if updated:
                    return {"status": "success", "message": "Zone updated"}
                return {"status": "info", "message": "No changes made"}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/api/zones/{zone_id}")
        async def delete_zone(zone_id: int,
                              current_user: dict = Depends(require_admin)):
            try:
                deleted = self.db.delete_zone(zone_id)
                self._refresh_zone_caches(force=True)
                if deleted:
                    return {"status": "success", "message": "Zone deleted"}
                raise HTTPException(status_code=404, detail="Zone not found")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # ── Camera registry endpoints ──────────────────────────────

        @self.app.get("/api/cameras-db")
        async def list_cameras_db(current_user: dict = Depends(get_current_user)):
            """List all cameras with their zone assignments."""
            try:
                cameras = self.db.list_cameras_db()
                return {"cameras": convert_to_serializable(cameras)}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/cameras-db")
        async def create_camera_entry(req: CameraCreateRequest,
                                       current_user: dict = Depends(require_admin)):
            try:
                camera_id = req.camera_id.strip()
                if not camera_id:
                    raise HTTPException(status_code=400, detail="camera_id is required")
                cid = self.db.upsert_camera(
                    camera_id=camera_id,
                    name=req.name,
                    zone_id=req.zone_id,
                    location=req.location,
                    stream_url=req.stream_url,
                    camera_type=req.camera_type or 'ip',
                )
                self._refresh_zone_caches(force=True)
                return {"status": "success", "id": cid, "camera_id": camera_id}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put("/api/cameras-db/{camera_id}")
        async def update_camera_entry(camera_id: str, req: CameraUpdateRequest,
                                       current_user: dict = Depends(require_admin)):
            try:
                zone_arg = ...
                if req.clear_zone:
                    zone_arg = None
                elif req.zone_id is not None:
                    zone_arg = req.zone_id
                updated = self.db.update_camera(
                    camera_id,
                    name=req.name,
                    zone_id=zone_arg,
                    location=req.location,
                    stream_url=req.stream_url,
                )
                self._refresh_zone_caches(force=True)
                if updated:
                    return {"status": "success", "message": "Camera updated"}
                raise HTTPException(status_code=404, detail="Camera not found or no changes")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/api/cameras-db/{camera_id}")
        async def delete_camera_entry(camera_id: str,
                                       current_user: dict = Depends(require_admin)):
            try:
                deleted = self.db.delete_camera(camera_id)
                self._refresh_zone_caches(force=True)
                if deleted:
                    return {"status": "success", "message": "Camera deleted"}
                raise HTTPException(status_code=404, detail="Camera not found")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # ── Alarm management endpoints ─────────────────────────────

        @self.app.get("/api/logs")
        async def list_detection_logs(
            type: Optional[str] = None,
            camera_id: Optional[str] = None,
            status: Optional[str] = None,
            search: Optional[str] = None,
            date_from: Optional[str] = None,
            date_to: Optional[str] = None,
            limit: int = 50,
            offset: int = 0,
            current_user: dict = Depends(get_current_user)
        ):
            """List detection logs with filters and pagination."""
            try:
                result = self.db.list_detection_logs(
                    log_type=type, camera_id=camera_id, status=status,
                    search=search, date_from=date_from, date_to=date_to,
                    limit=limit, offset=offset
                )
                return convert_to_serializable(result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/logs/stats")
        async def detection_log_stats(
            hours: int = 24,
            current_user: dict = Depends(get_current_user)
        ):
            """Detection log statistics for dashboard cards."""
            try:
                return convert_to_serializable(self.db.get_detection_log_stats(hours=hours))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/logs/timeseries")
        async def detection_log_timeseries(
            hours: int = 24,
            current_user: dict = Depends(get_current_user)
        ):
            """Time-bucketed detection counts for charting."""
            try:
                return convert_to_serializable(self.db.get_detection_log_timeseries(hours=hours))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/logs/breakdown")
        async def detection_log_breakdown(
            hours: int = 24,
            current_user: dict = Depends(get_current_user)
        ):
            """Per-camera / per-type / per-severity breakdowns."""
            try:
                return convert_to_serializable(self.db.get_detection_log_breakdown(hours=hours))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/logs/export")
        async def export_detection_logs(
            type: Optional[str] = None,
            camera_id: Optional[str] = None,
            status: Optional[str] = None,
            search: Optional[str] = None,
            date_from: Optional[str] = None,
            date_to: Optional[str] = None,
            current_user: dict = Depends(get_current_user)
        ):
            """Export filtered detection logs as CSV."""
            try:
                result = self.db.list_detection_logs(
                    log_type=type, camera_id=camera_id, status=status,
                    search=search, date_from=date_from, date_to=date_to,
                    limit=10000, offset=0
                )
                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow(['timestamp', 'type', 'camera_id', 'subject',
                                 'confidence', 'severity', 'status', 'details'])
                for log in result['logs']:
                    details = log.get('details')
                    if isinstance(details, (dict, list)):
                        details = json.dumps(details)
                    writer.writerow([
                        log.get('created_at').isoformat() if log.get('created_at') else '',
                        log.get('type', ''),
                        log.get('camera_id', ''),
                        log.get('subject', '') or '',
                        log.get('confidence', '') if log.get('confidence') is not None else '',
                        log.get('severity', '') or '',
                        log.get('status', '') or '',
                        details or ''
                    ])
                buf.seek(0)
                return StreamingResponse(
                    iter([buf.getvalue()]),
                    media_type='text/csv',
                    headers={'Content-Disposition': 'attachment; filename="detection_logs.csv"'}
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/alarms")
        async def list_alarms(
            status: Optional[str] = None,
            type: Optional[str] = None,
            severity: Optional[str] = None,
            camera_id: Optional[str] = None,
            limit: int = 50,
            offset: int = 0,
            current_user: dict = Depends(get_current_user)
        ):
            """List alarms with filters"""
            try:
                result = self.db.list_alarms(
                    status=status, alarm_type=type, severity=severity,
                    camera_id=camera_id, limit=limit, offset=offset
                )
                return convert_to_serializable(result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/alarms/stats")
        async def alarm_stats(current_user: dict = Depends(get_current_user)):
            """Get alarm statistics"""
            try:
                stats = self.db.get_alarm_stats()
                return convert_to_serializable(stats)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/alarms/{alarm_id}")
        async def get_alarm(alarm_id: int, current_user: dict = Depends(get_current_user)):
            """Get single alarm with full details including snapshot"""
            try:
                alarm = self.db.get_alarm(alarm_id)
                if not alarm:
                    raise HTTPException(status_code=404, detail="Alarm not found")
                return convert_to_serializable(alarm)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.patch("/api/alarms/{alarm_id}")
        async def update_alarm(alarm_id: int, req: UpdateAlarmRequest,
                               current_user: dict = Depends(get_current_user)):
            """Update alarm status/notes"""
            try:
                if req.status and req.status not in ('unresolved', 'resolved', 'false_alarm'):
                    raise HTTPException(status_code=400, detail="Invalid status")
                resolved_by = current_user.get("full_name") or current_user.get("sub", "")
                updated = self.db.update_alarm(
                    alarm_id=alarm_id, status=req.status,
                    notes=req.notes, resolved_by=resolved_by
                )
                if updated:
                    return {"status": "success", "message": "Alarm updated"}
                raise HTTPException(status_code=404, detail="Alarm not found")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/alarms/bulk-update")
        async def bulk_update_alarms(req: BulkUpdateAlarmsRequest,
                                     current_user: dict = Depends(get_current_user)):
            """Bulk update alarm statuses"""
            try:
                if req.status not in ('unresolved', 'resolved', 'false_alarm'):
                    raise HTTPException(status_code=400, detail="Invalid status")
                resolved_by = current_user.get("full_name") or current_user.get("sub", "")
                count = self.db.bulk_update_alarms(req.alarm_ids, req.status, resolved_by)
                return {"status": "success", "updated": count}
            except HTTPException:
                raise
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
                    except Exception:
                        break

                    await asyncio.sleep(0.03)  # ~30 FPS

            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                if websocket in self.client_connections:
                    self.client_connections.remove(websocket)
                logger.info(f"🔌 WebSocket disconnected. Total: {len(self.client_connections)}")
    
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
                    cap = self.video_capture
                    if self.is_streaming and cap is not None:
                        try:
                            opened = cap.isOpened()
                        except Exception:
                            opened = False
                        if opened:
                            try:
                                ret, frame = cap.read()
                            except Exception:
                                ret, frame = False, None
                            if ret and frame is not None and getattr(frame, 'size', 0) > 0:
                                captured_any = True
                                self.stats['frames_captured'] += 1
                                current_frame_id = self.frame_id
                                self.frame_id += 1

                                self.frame_counter += 1
                                if self.frame_counter % self.frame_skip == 0:
                                    self._dispatch_frame(current_frame_id, "CAM-01", frame)
                    
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
                        
                        # Re-check camera still exists (may have been removed)
                        with self.cameras_lock:
                            if cam_id not in self.cameras:
                                continue

                        try:
                            ret, frame = cap.read()
                        except Exception:
                            continue
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
            """Process frames for facial recognition. InsightFace + FER return
            age/gender/emotion inline with recognition results."""
            logger.info("👤 Thread 2: Face Recognition + Demographics started")

            while True:
                try:
                    frame_id, camera_id, frame = self.face_processing_queue.get(timeout=1.0)

                    start_time = time.time()

                    processed_frame, face_results = self._process_faces(frame)

                    if self.demographics_enabled and face_results:
                        for result in face_results:
                            if result.get('name') == 'Unknown':
                                self.stats['unknown_faces'] += 1

                            if result.get('age') is not None or result.get('gender') or result.get('emotion'):
                                self.stats['demographics_analyzed'] += 1
                                bbox = result.get('bbox')
                                if not bbox:
                                    continue
                                x, y, w, h = bbox
                                try:
                                    age = result.get('age')
                                    gender = result.get('gender')
                                    emotion = result.get('emotion')
                                    lines = []
                                    if age is not None:
                                        lines.append(f"Age: {age}")
                                    if gender:
                                        lines.append(f"{gender}")
                                    if emotion:
                                        lines.append(f"{emotion}")
                                    ty = max(0, y - 10)
                                    for i, line in enumerate(lines):
                                        cv2.putText(processed_frame, line,
                                                    (x, ty - i * 18),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                    (0, 255, 255), 1, cv2.LINE_AA)
                                except Exception as e:
                                    logger.debug(f"demographics draw failed: {e}")
                    
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
                        confirmed = [r for r in fire_results if r.get('confirmed', False)]
                        self.stats['fire_detections'] += len(confirmed)
                        if confirmed:
                            logger.debug(f"🔥 Confirmed {len(confirmed)} fire/smoke in {processing_time:.3f}s")

                        critical = [r for r in confirmed if r.get('severity') == 'critical']
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

                            # ── Persist detection logs (historical record) ──
                            try:
                                for r in (face_results or []):
                                    name = r.get('name') or 'Unknown'
                                    self._log_detection_if_needed(
                                        camera_id, 'face',
                                        subject=name,
                                        confidence=r.get('confidence'),
                                        severity='critical' if name == 'Unknown' else None,
                                        status='unknown' if name == 'Unknown' else 'recognized',
                                        details={
                                            'name': name,
                                            'employee_id': r.get('employee_id'),
                                            'age': r.get('age'),
                                            'gender': r.get('gender'),
                                            'emotion': r.get('emotion'),
                                            'bbox': r.get('bbox'),
                                        }
                                    )
                                for r in (plate_results or []):
                                    plate = r.get('plate') or r.get('plate_number')
                                    authorised = r.get('authorised')
                                    if authorised is None:
                                        authorised = r.get('is_authorized')
                                    self._log_detection_if_needed(
                                        camera_id, 'plate',
                                        subject=plate,
                                        confidence=r.get('confidence'),
                                        status=('authorized' if authorised else 'unauthorized') if authorised is not None else None,
                                        details={
                                            'plate': plate,
                                            'owner': r.get('owner'),
                                            'vehicle_type': r.get('vehicle_type'),
                                            'authorised': authorised,
                                        }
                                    )
                                for r in (fire_results or []):
                                    cls = r.get('class') or 'fire'
                                    self._log_detection_if_needed(
                                        camera_id, 'fire',
                                        subject=cls,
                                        confidence=r.get('confidence'),
                                        severity=r.get('severity', 'high'),
                                        status='alert' if r.get('alert') else 'detected',
                                        details={
                                            'class': cls,
                                            'area_ratio': r.get('area_ratio'),
                                            'alert': r.get('alert'),
                                        }
                                    )
                                for r in (har_results or []):
                                    cls = r.get('class') or 'action'
                                    if cls == 'normal':
                                        continue
                                    self._log_detection_if_needed(
                                        camera_id, 'har',
                                        subject=r.get('action_label') or cls,
                                        confidence=r.get('confidence'),
                                        severity=r.get('severity'),
                                        status=r.get('severity'),
                                        details={
                                            'action': cls,
                                            'action_label': r.get('action_label'),
                                        }
                                    )
                                for r in (weapon_results or []):
                                    cls = r.get('class') or 'weapon'
                                    self._log_detection_if_needed(
                                        camera_id, 'weapon',
                                        subject=cls,
                                        confidence=r.get('confidence'),
                                        severity=r.get('severity', 'critical'),
                                        status='threat',
                                        details={'class': cls}
                                    )
                            except Exception as _log_err:
                                logger.error(f"Detection log write failed: {_log_err}")

                            # ── Create alarms for detections ──────
                            # Capture a snapshot for alarms (low quality to save space)
                            snapshot_b64 = None
                            has_alarm = False

                            # Check if any alarm-worthy detections exist
                            if face_results:
                                for r in face_results:
                                    if r.get('name') == 'Unknown':
                                        has_alarm = True
                                        break
                            if not has_alarm and (fire_results or
                                    [r for r in (har_results or []) if r.get('class') != 'normal'] or
                                    weapon_results):
                                has_alarm = True

                            # Also flag when any face (unknown or recognized-but-unauthorized)
                            # appears in a restricted zone, so snapshot is captured.
                            if not has_alarm and face_results:
                                zone_info = self._get_camera_zone_info(camera_id)
                                if zone_info and zone_info.get('is_restricted'):
                                    for r in face_results:
                                        name = r.get('name') or 'Unknown'
                                        emp = r.get('employee_id')
                                        if name == 'Unknown' or not emp or \
                                           not self._person_authorized_for_zone(emp, zone_info['zone_name']):
                                            has_alarm = True
                                            break

                            if has_alarm:
                                try:
                                    snap_frame = final_frame
                                    h, w = snap_frame.shape[:2]
                                    if w > 400:
                                        scale = 400 / w
                                        snap_frame = cv2.resize(snap_frame, (400, int(h * scale)))
                                    _, snap_buf = cv2.imencode('.jpg', snap_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                                    snapshot_b64 = base64.b64encode(snap_buf).decode('utf-8')
                                except Exception:
                                    snapshot_b64 = None

                            # Face alarms (unknown persons)
                            if face_results:
                                for r in face_results:
                                    if r.get('name') == 'Unknown':
                                        self._create_alarm_if_needed(
                                            camera_id, 'face', 'critical',
                                            f"Unauthorized person detected ({(r.get('confidence', 0) * 100):.0f}% confidence)",
                                            snapshot_b64,
                                            {'confidence': r.get('confidence'), 'bbox': r.get('bbox')}
                                        )

                            # Zone authorization check: recognized person in restricted zone
                            try:
                                self._check_zone_authorization(camera_id, face_results, snapshot_b64)
                            except Exception as _zerr:
                                logger.error(f"Zone authorization check failed: {_zerr}")

                            # Fire alarms
                            if fire_results:
                                for r in fire_results:
                                    sev = r.get('severity', 'high')
                                    cls = r.get('class', 'fire').upper()
                                    self._create_alarm_if_needed(
                                        camera_id, 'fire', sev,
                                        f"{cls} detected ({(r.get('confidence', 0) * 100):.0f}% confidence)",
                                        snapshot_b64,
                                        {'class': r.get('class'), 'confidence': r.get('confidence'),
                                         'area_ratio': r.get('area_ratio')}
                                    )

                            # HAR alarms (non-normal actions)
                            if har_results:
                                for r in har_results:
                                    if r.get('class') != 'normal':
                                        sev = r.get('severity', 'high')
                                        label = r.get('action_label', r.get('class', 'ACTION')).upper()
                                        self._create_alarm_if_needed(
                                            camera_id, 'har', sev,
                                            f"{label} detected ({(r.get('confidence', 0) * 100):.0f}% confidence)",
                                            snapshot_b64,
                                            {'action': r.get('class'), 'confidence': r.get('confidence')}
                                        )

                            # Weapon alarms
                            if weapon_results:
                                for r in weapon_results:
                                    sev = r.get('severity', 'critical')
                                    cls = (r.get('class') or 'WEAPON').upper()
                                    self._create_alarm_if_needed(
                                        camera_id, 'weapon', sev,
                                        f"{cls} detected ({(r.get('confidence', 0) * 100):.0f}% confidence)",
                                        snapshot_b64,
                                        {'class': r.get('class'), 'confidence': r.get('confidence')}
                                    )

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
    
    def _log_detection_if_needed(self, camera_id: str, log_type: str,
                                 subject: str = None, confidence: float = None,
                                 severity: str = None, status: str = None,
                                 details: Dict = None):
        """Persist a detection to detection_logs with per-subject cooldown."""
        cooldown_key = f"{camera_id}:{log_type}:{subject or ''}"
        current_time = time.time()
        with self.log_lock:
            last_time = self.log_cooldowns.get(cooldown_key, 0)
            if current_time - last_time < self.log_cooldown_seconds:
                return
            self.log_cooldowns[cooldown_key] = current_time
            # Opportunistic cleanup to bound dict size
            if len(self.log_cooldowns) > 2000:
                cutoff = current_time - (self.log_cooldown_seconds * 10)
                self.log_cooldowns = {k: v for k, v in self.log_cooldowns.items() if v > cutoff}
        try:
            self.db.insert_detection_log(
                camera_id=camera_id, log_type=log_type, subject=subject,
                confidence=float(confidence) if confidence is not None else None,
                severity=severity, status=status,
                details=convert_to_serializable(details) if details else None
            )
        except Exception as e:
            logger.error(f"Error inserting detection log: {e}")

    def _create_alarm_if_needed(self, camera_id: str, alarm_type: str,
                                severity: str, description: str,
                                snapshot_base64: str = None,
                                detection_metadata: Dict = None):
        """Create an alarm in the database with deduplication (cooldown-based)"""
        cooldown_key = f"{camera_id}:{alarm_type}"
        current_time = time.time()

        with self.alarm_lock:
            last_time = self.alarm_cooldowns.get(cooldown_key, 0)
            if current_time - last_time < self.alarm_cooldown_seconds:
                return  # Skip — duplicate within cooldown
            self.alarm_cooldowns[cooldown_key] = current_time

        try:
            self.db.create_alarm(
                camera_id=camera_id,
                alarm_type=alarm_type,
                severity=severity,
                description=description,
                snapshot=snapshot_base64,
                detection_metadata=convert_to_serializable(detection_metadata) if detection_metadata else None
            )
            logger.info(f"🚨 Alarm created: [{severity.upper()}] {alarm_type} on {camera_id} — {description}")
        except Exception as e:
            logger.error(f"Error creating alarm: {e}")

    def _refresh_zone_caches(self, force: bool = False):
        """Refresh camera→zone and person→zones caches (TTL-based)."""
        now = time.time()
        with self._zone_cache_lock:
            if not force and (now - self._zone_cache_refreshed_at) < self._zone_cache_ttl:
                return
            try:
                cameras_db = self.db.list_cameras_db()
                cam_map = {}
                for c in cameras_db:
                    if c.get('zone_id') is not None:
                        cam_map[c['camera_id']] = {
                            'zone_id': c['zone_id'],
                            'zone_name': c['zone_name'],
                            'is_restricted': bool(c['zone_is_restricted']),
                        }
                    else:
                        cam_map[c['camera_id']] = None
                self._camera_zone_cache = cam_map

                self.db.cursor.execute(
                    "SELECT employee_id, authorized_zones FROM persons"
                )
                person_map = {}
                for row in self.db.cursor.fetchall():
                    emp = row.get('employee_id')
                    zones = row.get('authorized_zones') or []
                    if emp:
                        person_map[emp] = set(zones)
                self._person_zones_cache = person_map
                self._zone_cache_refreshed_at = now
            except Exception as e:
                logger.error(f"Error refreshing zone caches: {e}")

    def _get_camera_zone_info(self, camera_id: str):
        """Return zone info dict for a camera or None."""
        self._refresh_zone_caches()
        with self._zone_cache_lock:
            if camera_id in self._camera_zone_cache:
                return self._camera_zone_cache[camera_id]
        # Not cached — try a direct DB lookup as fallback
        try:
            info = self.db.get_camera_zone(camera_id)
            if info:
                info = {
                    'zone_id': info['zone_id'],
                    'zone_name': info['zone_name'],
                    'is_restricted': bool(info['is_restricted']),
                }
                with self._zone_cache_lock:
                    self._camera_zone_cache[camera_id] = info
                return info
        except Exception:
            pass
        return None

    def _person_authorized_for_zone(self, employee_id: str, zone_name: str) -> bool:
        if not employee_id or not zone_name:
            return False
        self._refresh_zone_caches()
        with self._zone_cache_lock:
            zones = self._person_zones_cache.get(employee_id)
        if zones is not None:
            return zone_name in zones
        # Fallback direct lookup
        try:
            person = self.db.get_person_by_employee_id(employee_id)
            if person:
                authorized = person.get('authorized_zones') or []
                return zone_name in authorized
        except Exception:
            pass
        return False

    def _check_zone_authorization(self, camera_id: str, face_results: List[Dict],
                                  snapshot_b64: str = None):
        """Emit 'unauthorized_zone' alarms when a person is seen in a restricted
        zone without being listed in their authorized_zones."""
        if not face_results:
            return
        zone_info = self._get_camera_zone_info(camera_id)
        if not zone_info or not zone_info.get('is_restricted'):
            return
        zone_name = zone_info['zone_name']
        for r in face_results:
            name = r.get('name') or 'Unknown'
            emp_id = r.get('employee_id')
            confidence = r.get('confidence', 0) or 0
            is_unknown = (name == 'Unknown') or not emp_id
            if not is_unknown and self._person_authorized_for_zone(emp_id, zone_name):
                # Recognized AND authorized — no alarm
                continue
            if is_unknown:
                description = (
                    f"Unknown person in restricted zone '{zone_name}' "
                    f"({(confidence * 100):.0f}% confidence)"
                )
                subject = f"Unknown @ {zone_name}"
            else:
                description = (
                    f"{name} ({emp_id}) in restricted zone '{zone_name}' "
                    f"— not authorized ({(confidence * 100):.0f}% confidence)"
                )
                subject = f"{name} @ {zone_name}"

            self._create_alarm_if_needed(
                camera_id, 'unauthorized_zone', 'critical',
                description, snapshot_b64,
                {
                    'zone_name': zone_name,
                    'zone_id': zone_info.get('zone_id'),
                    'person_name': name,
                    'employee_id': emp_id,
                    'confidence': confidence,
                    'bbox': r.get('bbox'),
                }
            )
            try:
                self._log_detection_if_needed(
                    camera_id, 'unauthorized_zone',
                    subject=subject,
                    confidence=confidence, severity='critical',
                    status='unauthorized',
                    details={
                        'zone_name': zone_name,
                        'person_name': name,
                        'employee_id': emp_id,
                    }
                )
            except Exception:
                pass

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
        pass


# Create the API instance
security_api = EnhancedSecuritySystemAPI()
app = security_api.app

if __name__ == "__main__":
    import uvicorn
    logger.info("🔒 Starting Enhanced Security System API v4.0.0...")
    logger.info("📡 API: http://localhost:8000")
    logger.info("🔌 WebSocket: ws://localhost:8000/ws")
    logger.info("📚 Docs: http://localhost:8000/docs")
    logger.info("🧵 Multi-threaded processing: 7 threads")
    logger.info("   - Thread 1: Video Capture")
    logger.info("   - Thread 2: Facial Recognition + Demographics (InsightFace + FER)")
    logger.info("   - Thread 3: License Plate Recognition")
    logger.info("   - Thread 4: Fire & Smoke Detection")
    logger.info("   - Thread 5: Human Action Recognition (SlowFast)")
    logger.info("   - Thread 6: Weapon Detection")
    logger.info("   - Thread 7: Results Merging")

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