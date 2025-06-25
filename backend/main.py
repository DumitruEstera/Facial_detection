"""
Updated FastAPI backend with PostgreSQL database integration
for the Military Security Facial Recognition System
"""

from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import cv2
import base64
import asyncio
import json
import sys
import os
import numpy as np
from dotenv import load_dotenv
from typing import Optional
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import our database-integrated facial recognition system
try:
    from database_facial_recognition import DatabaseFacialRecognitionSystem
    from database_manager import DatabaseManager
    FACIAL_RECOGNITION_AVAILABLE = True
    print("✅ Database facial recognition module loaded successfully!")
except ImportError as e:
    print(f"❌ Error importing database facial recognition: {e}")
    print("ℹ️  Running in demo mode without facial recognition")
    FACIAL_RECOGNITION_AVAILABLE = False

# Global variables
fr_system = None
db_manager = None
system_active = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global fr_system, db_manager
    
    if FACIAL_RECOGNITION_AVAILABLE:
        try:
            # Initialize facial recognition system
            fr_system = DatabaseFacialRecognitionSystem()
            
            # Initialize database
            db_success = await fr_system.initialize_database()
            
            if db_success:
                db_manager = fr_system.db_manager
                print("✅ Database facial recognition system initialized successfully!")
            else:
                print("❌ Database initialization failed, running in demo mode")
                fr_system = None
                
        except Exception as e:
            print(f"❌ Error initializing database facial recognition system: {e}")
            print("ℹ️  Running in demo mode")
            fr_system = None
    else:
        print("ℹ️  Starting in demo mode - no facial recognition available")
    
    yield
    
    # Shutdown
    if fr_system:
        await fr_system.cleanup()
        print("✅ Database facial recognition system cleaned up")

app = FastAPI(
    title="Military Security API with Database Integration", 
    version="2.0.0", 
    lifespan=lifespan
)

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Military Security API with Database Integration", 
        "version": "2.0.0",
        "facial_recognition_available": FACIAL_RECOGNITION_AVAILABLE,
        "database_available": fr_system.db_initialized if fr_system else False,
        "system_active": system_active
    }

@app.get("/api/status")
async def get_status():
    """Get system status including database information"""
    status = {
        "system_active": system_active,
        "facial_recognition_available": FACIAL_RECOGNITION_AVAILABLE,
        "database_available": fr_system.db_initialized if fr_system else False,
        "camera_available": fr_system.camera_available if fr_system else False,
        "known_faces": len(fr_system.known_face_names) if fr_system else 0
    }
    
    if db_manager:
        try:
            # Test database connection
            db_test = await db_manager.test_connection()
            status["database_connection"] = db_test
        except Exception as e:
            status["database_connection"] = False
            status["database_error"] = str(e)
    
    return status

@app.get("/api/personnel")
async def get_personnel(limit: int = Query(50, ge=1, le=100)):
    """Get personnel list from database"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # This would need to be implemented in DatabaseManager
        # For now, return basic info
        return {
            "total": len(fr_system.known_face_names) if fr_system else 0,
            "personnel": [
                {"id": i, "name": name} 
                for i, name in enumerate(fr_system.known_face_names[:limit])
            ] if fr_system else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving personnel: {str(e)}")

@app.get("/api/alerts")
async def get_recent_alerts(limit: int = Query(10, ge=1, le=50)):
    """Get recent security alerts"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        alerts = await db_manager.get_recent_alerts(limit)
        return {"alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving alerts: {str(e)}")

@app.post("/api/start-system")
async def start_system():
    global system_active
    
    if not FACIAL_RECOGNITION_AVAILABLE or not fr_system:
        raise HTTPException(status_code=503, detail="Facial recognition not available")
    
    if not fr_system.db_initialized:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    if not fr_system.camera_available:
        raise HTTPException(status_code=503, detail="Camera not available")
    
    system_active = True
    
    # Log system start in database
    if db_manager:
        try:
            await db_manager.create_security_alert(
                alert_type="System Status",
                severity="LOW",
                description="Facial recognition system started",
                additional_data={"action": "system_start", "timestamp": datetime.now().isoformat()}
            )
        except Exception as e:
            logger.error(f"Failed to log system start: {e}")
    
    return {"status": "started", "message": "Database-integrated system activated successfully"}

@app.post("/api/stop-system")
async def stop_system():
    global system_active
    system_active = False
    
    # Log system stop in database
    if db_manager:
        try:
            await db_manager.create_security_alert(
                alert_type="System Status",
                severity="LOW",
                description="Facial recognition system stopped",
                additional_data={"action": "system_stop", "timestamp": datetime.now().isoformat()}
            )
        except Exception as e:
            logger.error(f"Failed to log system stop: {e}")
    
    return {"status": "stopped", "message": "System deactivated successfully"}

@app.post("/api/register-face")
async def register_face(
    first_name: str = Query(..., description="First name"),
    last_name: str = Query(..., description="Last name"),
    rank: str = Query("STUDENT", description="Military rank")
):
    """Register a new face with the given name using database storage"""
    if not FACIAL_RECOGNITION_AVAILABLE or not fr_system:
        raise HTTPException(status_code=503, detail="Facial recognition not available")
    
    if not fr_system.db_initialized:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    if not system_active:
        raise HTTPException(status_code=400, detail="System is not active")
    
    first_name = first_name.strip()
    last_name = last_name.strip()
    
    if not first_name or not last_name:
        raise HTTPException(status_code=400, detail="First name and last name cannot be empty")
    
    try:
        # Check if camera is available
        if not fr_system.camera_available:
            raise HTTPException(status_code=503, detail="Camera not available")
            
        # Capture current frame
        ret, frame = fr_system.video_capture.read()
        if not ret:
            raise HTTPException(status_code=500, detail="Could not capture frame")
        
        # Detect faces in current frame
        recognized_faces = await fr_system.recognize_faces_db(frame)
        
        if not recognized_faces:
            raise HTTPException(status_code=400, detail="No face detected in current frame. Position yourself in front of the camera.")
        
        # Use the first detected face
        face_bbox = recognized_faces[0]['bbox']
        
        # Register the face in database
        success, message = await fr_system.register_new_face_db(
            frame, face_bbox, first_name, last_name, rank
        )
        
        if success:
            return {
                "status": "success", 
                "message": message,
                "total_known_faces": len(fr_system.known_face_names)
            }
        else:
            raise HTTPException(status_code=500, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in register_face: {e}")
        raise HTTPException(status_code=500, detail=f"Error registering face: {str(e)}")

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")
    
    try:
        while True:
            if (FACIAL_RECOGNITION_AVAILABLE and fr_system and 
                fr_system.db_initialized and system_active and 
                fr_system.camera_available):
                try:
                    # Capture frame from facial recognition system
                    ret, frame = fr_system.video_capture.read()
                    if ret:
                        # Process frame with database-integrated facial recognition
                        recognized_faces = await fr_system.recognize_faces_db(frame)
                        
                        # Convert any remaining numpy types to native Python types
                        recognized_faces = convert_numpy_types(recognized_faces)
                        
                        # Draw face boxes
                        frame = fr_system.draw_face_boxes(frame, recognized_faces)
                        
                        # Encode frame to base64
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        frame_data = base64.b64encode(buffer).decode('utf-8')
                        
                        # Send frame and face data
                        message = {
                            "frame": frame_data,
                            "faces": recognized_faces,
                            "timestamp": asyncio.get_event_loop().time(),
                            "system_active": system_active,
                            "database_active": True
                        }
                        
                        # Ensure all data is JSON serializable
                        message = convert_numpy_types(message)
                        
                        await websocket.send_text(json.dumps(message))
                        
                    else:
                        # Send error if frame capture failed
                        await websocket.send_text(json.dumps({
                            "error": "Failed to capture frame from camera",
                            "faces": [],
                            "timestamp": asyncio.get_event_loop().time(),
                            "system_active": system_active,
                            "database_active": True
                        }))
                        
                except Exception as e:
                    # Send error message
                    logger.error(f"Error processing video frame: {e}")
                    await websocket.send_text(json.dumps({
                        "error": f"Processing error: {str(e)}",
                        "faces": [],
                        "timestamp": asyncio.get_event_loop().time(),
                        "system_active": system_active,
                        "database_active": fr_system.db_initialized if fr_system else False
                    }))
            else:
                # Send demo data when system is not active or not available
                status_message = "Demo mode - no real video feed"
                if not FACIAL_RECOGNITION_AVAILABLE:
                    status_message = "Facial recognition not available"
                elif not fr_system:
                    status_message = "Facial recognition system not initialized"
                elif not fr_system.db_initialized:
                    status_message = "Database not initialized"
                elif not fr_system.camera_available:
                    status_message = "Camera not available"
                elif not system_active:
                    status_message = "System not active"
                
                await websocket.send_text(json.dumps({
                    "demo_mode": True,
                    "message": status_message,
                    "faces": [],
                    "timestamp": asyncio.get_event_loop().time(),
                    "system_active": system_active,
                    "database_active": fr_system.db_initialized if fr_system else False
                }))
                
            await asyncio.sleep(0.1)  # ~10 FPS
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Don't try to close if already disconnected
        try:
            if websocket.client_state.name != "DISCONNECTED":
                await websocket.close()
        except:
            pass  # Connection already closed
        logger.info("WebSocket connection closed")

@app.get("/api/database-test")
async def test_database():
    """Test database connection and return diagnostics"""
    if not db_manager:
        return {"status": "error", "message": "Database manager not available"}
    
    try:
        # Test basic connection
        connection_ok = await db_manager.test_connection()
        
        if not connection_ok:
            return {"status": "error", "message": "Database connection failed"}
        
        # Get some statistics
        face_count = len(fr_system.known_face_names) if fr_system else 0
        
        return {
            "status": "success", 
            "message": "Database connection successful",
            "known_faces": face_count,
            "database_config": {
                "host": db_manager.db_config['host'],
                "port": db_manager.db_config['port'],
                "database": db_manager.db_config['database'],
                "user": db_manager.db_config['user']
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"Database test failed: {str(e)}"}

@app.get("/api/camera-test")
async def test_camera():
    """Test if camera is working"""
    if not FACIAL_RECOGNITION_AVAILABLE or not fr_system:
        return {"status": "error", "message": "Facial recognition not available"}
    
    if not fr_system.camera_available:
        return {"status": "error", "message": "Camera not available"}
    
    try:
        ret, frame = fr_system.video_capture.read()
        if ret:
            height, width = frame.shape[:2]
            return {
                "status": "success", 
                "message": "Camera is working",
                "resolution": f"{width}x{height}",
                "database_integrated": fr_system.db_initialized
            }
        else:
            return {"status": "error", "message": "Could not read from camera"}
    except Exception as e:
        return {"status": "error", "message": f"Camera test failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Military Security API with Database Integration...")
    print(f"📁 Current directory: {current_dir}")
    print(f"🔍 Facial recognition available: {FACIAL_RECOGNITION_AVAILABLE}")
    print("🌐 API will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    print("🗄️  Using PostgreSQL database with pgvector")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")