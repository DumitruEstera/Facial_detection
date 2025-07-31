#!/usr/bin/env python3
"""
FastAPI Backend for Security System
Integrates facial recognition and license plate recognition
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import cv2
import asyncio
import json
import base64
import threading
import queue
import time
import numpy as np
from datetime import datetime

# Import your existing modules
from facial_recognition_system import FacialRecognitionSystem
from license_plate_recognition_system import LicensePlateRecognitionSystem
from database_manager import DatabaseManager

# Pydantic models for API requests
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
    'password': 'admin'  # Change this to your PostgreSQL password
}

def convert_to_serializable(obj):
    """
    Recursively convert numpy types and other non-serializable types to Python native types
    """
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
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    else:
        return obj

class SecuritySystemAPI:
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
        self.current_mode = "both"  # "face", "plate", "both"
        self.frame_queue = queue.Queue(maxsize=10)
        self.client_connections: List[WebSocket] = []
        
        # Start background tasks
        self.setup_background_tasks()
        
    def setup_cors(self):
        """Setup CORS for React frontend"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000"],  # React dev server
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
            """Get system status"""
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
            """Start camera streaming"""
            try:
                if not self.is_streaming:
                    self.video_capture = cv2.VideoCapture(camera_id)
                    if self.video_capture.isOpened():
                        self.is_streaming = True
                        print(f"✅ Camera {camera_id} started successfully")
                        return {"status": "success", "message": "Camera started"}
                    else:
                        raise HTTPException(status_code=400, detail="Could not open camera")
                else:
                    return {"status": "info", "message": "Camera already running"}
            except Exception as e:
                print(f"❌ Error starting camera: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/camera/stop")
        async def stop_camera():
            """Stop camera streaming"""
            try:
                if self.is_streaming:
                    self.is_streaming = False
                    if self.video_capture:
                        self.video_capture.release()
                    print("🛑 Camera stopped")
                    return {"status": "success", "message": "Camera stopped"}
                else:
                    return {"status": "info", "message": "Camera not running"}
            except Exception as e:
                print(f"❌ Error stopping camera: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/camera/mode")
        async def set_mode(request: dict):
            """Set detection mode: face, plate, or both"""
            mode = request.get("mode")
            if mode not in ["face", "plate", "both"]:
                raise HTTPException(status_code=400, detail="Invalid mode")
            self.current_mode = mode
            print(f"🔄 Detection mode set to: {mode}")
            return {"status": "success", "mode": mode}
        
        @self.app.post("/api/persons/register")
        async def register_person(person: PersonRegistration):
            """Register a new person"""
            try:
                # For now, return success - actual face capture would be done via webclient
                return {
                    "status": "success", 
                    "message": f"Person {person.name} registered successfully",
                    "person": person.dict()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/plates/register")
        async def register_plate(plate: LicensePlateRegistration):
            """Register a new license plate"""
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
        
        @self.app.get("/api/logs/face")
        async def get_face_logs(limit: int = 50):
            """Get recent face recognition logs"""
            try:
                # Implement this method in your DatabaseManager if needed
                return {"logs": [], "count": 0}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/logs/vehicle")
        async def get_vehicle_logs(limit: int = 50):
            """Get recent vehicle access logs"""
            try:
                logs = self.db.get_vehicle_access_logs(limit=limit)
                return {"logs": convert_to_serializable(logs), "count": len(logs)}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/statistics")
        async def get_statistics():
            """Get system statistics"""
            try:
                stats = self.db.get_statistics()
                return convert_to_serializable(stats)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.client_connections.append(websocket)
            print(f"🔌 WebSocket client connected. Total clients: {len(self.client_connections)}")
            
            try:
                while True:
                    # Check for new frames to send
                    try:
                        message = self.frame_queue.get(block=False)
                        await websocket.send_text(json.dumps(message))
                    except queue.Empty:
                        pass
                    except Exception as e:
                        print(f"❌ Error sending WebSocket message: {e}")
                        break
                    
                    # Keep connection alive
                    await asyncio.sleep(0.01)
                    
            except WebSocketDisconnect:
                if websocket in self.client_connections:
                    self.client_connections.remove(websocket)
                print(f"🔌 WebSocket client disconnected. Total clients: {len(self.client_connections)}")
            except Exception as e:
                print(f"❌ WebSocket error: {e}")
                if websocket in self.client_connections:
                    self.client_connections.remove(websocket)
    
    def setup_background_tasks(self):
        """Setup background tasks for video processing"""
        
        def video_processing_thread():
            """Background thread for video processing"""
            print("🎥 Video processing thread started")
            
            while True:
                try:
                    if self.is_streaming and self.video_capture and self.video_capture.isOpened():
                        ret, frame = self.video_capture.read()
                        if ret:
                            # Process frame based on current mode
                            face_results = []
                            plate_results = []
                            processed_frame = frame.copy()
                            
                            try:
                                if self.current_mode in ["face", "both"]:
                                    processed_frame, face_results = self.face_system.process_frame(frame)
                                    
                                if self.current_mode in ["plate", "both"]:
                                    plate_results = self.plate_system.process_frame(frame)
                                    if plate_results:
                                        processed_frame = self.plate_system.draw_outputs(processed_frame, plate_results)
                                
                                # Convert frame to base64 for WebSocket transmission
                                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                                
                                # Send to all connected clients
                                if self.client_connections:
                                    # Convert all results to serializable format
                                    serialized_face_results = convert_to_serializable(face_results)
                                    serialized_plate_results = convert_to_serializable(plate_results)
                                    
                                    message = {
                                        "type": "video_frame",
                                        "frame": frame_base64,
                                        "face_results": serialized_face_results,
                                        "plate_results": serialized_plate_results,
                                        "timestamp": datetime.now().isoformat()
                                    }
                                    
                                    # Store message for async sending
                                    try:
                                        self.frame_queue.put(message, block=False)
                                    except queue.Full:
                                        # Clear old frames if queue is full
                                        try:
                                            self.frame_queue.get(block=False)
                                            self.frame_queue.put(message, block=False)
                                        except queue.Empty:
                                            pass
                                        
                            except Exception as e:
                                print(f"❌ Error processing frame: {e}")
                                # Continue with unprocessed frame
                                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                                
                                if self.client_connections:
                                    message = {
                                        "type": "video_frame",
                                        "frame": frame_base64,
                                        "face_results": [],
                                        "plate_results": [],
                                        "timestamp": datetime.now().isoformat()
                                    }
                                    
                                    try:
                                        self.frame_queue.put(message, block=False)
                                    except queue.Full:
                                        pass
                    
                    time.sleep(0.033)  # ~30 FPS
                    
                except Exception as e:
                    print(f"❌ Error in video processing thread: {e}")
                    time.sleep(1)
        
        # Start video processing in background thread
        video_thread = threading.Thread(target=video_processing_thread, daemon=True)
        video_thread.start()

# Create the API instance
security_api = SecuritySystemAPI()
app = security_api.app

if __name__ == "__main__":
    import uvicorn
    print("🔒 Starting Security System API...")
    print("📡 API will be available at: http://localhost:8000")
    print("🔌 WebSocket endpoint: ws://localhost:8000/ws")
    print("📚 API documentation: http://localhost:8000/docs")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "app:app",  # Use string import for reload functionality
            host="0.0.0.0", 
            port=8000, 
            reload=False,  # Set to False for production
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down API server...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("Try running: uvicorn app:app --host 0.0.0.0 --port 8000")