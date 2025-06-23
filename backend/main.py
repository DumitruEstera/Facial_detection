from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import cv2
import base64
import asyncio
import json
import sys
import os
import numpy as np

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

# Add the parent directory to the Python path to find facial_recognition module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
facial_recognition_path = os.path.join(parent_dir, 'facial_recognition')
sys.path.append(facial_recognition_path)

# Now import the facial recognition system
try:
    from api_facial_recognition import APIFacialRecognitionSystem
    FACIAL_RECOGNITION_AVAILABLE = True
    print("✅ Facial recognition module loaded successfully!")
except ImportError as e:
    print(f"❌ Error importing facial recognition: {e}")
    print("ℹ️  Running in demo mode without facial recognition")
    FACIAL_RECOGNITION_AVAILABLE = False

# Global variables
fr_system = None
system_active = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global fr_system
    if FACIAL_RECOGNITION_AVAILABLE:
        try:
            fr_system = APIFacialRecognitionSystem()
            print("✅ Facial recognition system initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing facial recognition system: {e}")
            print("ℹ️  Running in demo mode")
    else:
        print("ℹ️  Starting in demo mode - no facial recognition available")
    
    yield
    
    # Shutdown
    if fr_system:
        fr_system.cleanup()
        print("✅ Facial recognition system cleaned up")

app = FastAPI(title="Military Security API", version="1.0.0", lifespan=lifespan)

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
        "message": "Military Security API", 
        "version": "1.0.0",
        "facial_recognition_available": FACIAL_RECOGNITION_AVAILABLE,
        "system_active": system_active
    }

@app.get("/api/status")
async def get_status():
    return {
        "system_active": system_active,
        "facial_recognition_available": FACIAL_RECOGNITION_AVAILABLE,
        "known_faces": len(fr_system.known_face_names) if fr_system else 0,
        "camera_available": fr_system.camera_available if fr_system else False
    }

@app.post("/api/start-system")
async def start_system():
    global system_active
    
    if not FACIAL_RECOGNITION_AVAILABLE or not fr_system:
        raise HTTPException(status_code=503, detail="Facial recognition not available")
    
    if not fr_system.camera_available:
        raise HTTPException(status_code=503, detail="Camera not available")
    
    system_active = True
    return {"status": "started", "message": "System activated successfully"}

@app.post("/api/stop-system")
async def stop_system():
    global system_active
    system_active = False
    return {"status": "stopped", "message": "System deactivated successfully"}

@app.post("/api/register-face")
async def register_face(name: str):
    """Register a new face with the given name"""
    if not FACIAL_RECOGNITION_AVAILABLE or not fr_system:
        raise HTTPException(status_code=503, detail="Facial recognition not available")
    
    if not system_active:
        raise HTTPException(status_code=400, detail="System is not active")
    
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    
    try:
        # Check if camera is available
        if not fr_system.camera_available:
            raise HTTPException(status_code=503, detail="Camera not available")
            
        # Capture current frame
        ret, frame = fr_system.video_capture.read()
        if not ret:
            raise HTTPException(status_code=500, detail="Could not capture frame")
        
        # Detect faces in current frame
        recognized_faces = fr_system.recognize_faces(frame)
        
        if not recognized_faces:
            raise HTTPException(status_code=400, detail="No face detected in current frame. Position yourself in front of the camera.")
        
        # Use the first detected face and ensure bbox is in proper format
        face_bbox = recognized_faces[0]['bbox']
        face_bbox = [int(coord) for coord in face_bbox]  # Convert to regular Python ints
        
        # Register the face
        success, message = fr_system.register_new_face_api(frame, face_bbox, name)
        
        if success:
            fr_system.save_known_faces()
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
        raise HTTPException(status_code=500, detail=f"Error registering face: {str(e)}")

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")
    
    try:
        while True:
            if FACIAL_RECOGNITION_AVAILABLE and fr_system and system_active and fr_system.camera_available:
                try:
                    # Capture frame from your facial recognition system
                    ret, frame = fr_system.video_capture.read()
                    if ret:
                        # Process frame with facial recognition
                        recognized_faces = fr_system.recognize_faces(frame)
                        
                        # Convert any remaining numpy types to native Python types
                        recognized_faces = convert_numpy_types(recognized_faces)
                        
                        frame = fr_system.draw_face_boxes(frame, recognized_faces)
                        
                        # Encode frame to base64
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        frame_data = base64.b64encode(buffer).decode('utf-8')
                        
                        # Send frame and face data
                        message = {
                            "frame": frame_data,
                            "faces": recognized_faces,
                            "timestamp": asyncio.get_event_loop().time(),
                            "system_active": system_active
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
                            "system_active": system_active
                        }))
                        
                except Exception as e:
                    # Send error message
                    print(f"Error processing video frame: {e}")
                    await websocket.send_text(json.dumps({
                        "error": f"Processing error: {str(e)}",
                        "faces": [],
                        "timestamp": asyncio.get_event_loop().time(),
                        "system_active": system_active
                    }))
            else:
                # Send demo data when system is not active or not available
                status_message = "Demo mode - no real video feed"
                if not FACIAL_RECOGNITION_AVAILABLE:
                    status_message = "Facial recognition not available"
                elif not fr_system:
                    status_message = "Facial recognition system not initialized"
                elif not fr_system.camera_available:
                    status_message = "Camera not available"
                elif not system_active:
                    status_message = "System not active"
                
                await websocket.send_text(json.dumps({
                    "demo_mode": True,
                    "message": status_message,
                    "faces": [],
                    "timestamp": asyncio.get_event_loop().time(),
                    "system_active": system_active
                }))
                
            await asyncio.sleep(0.1)  # ~10 FPS
            
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Don't try to close if already disconnected
        try:
            if websocket.client_state.name != "DISCONNECTED":
                await websocket.close()
        except:
            pass  # Connection already closed
        print("WebSocket connection closed")

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
                "resolution": f"{width}x{height}"
            }
        else:
            return {"status": "error", "message": "Could not read from camera"}
    except Exception as e:
        return {"status": "error", "message": f"Camera test failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Military Security API...")
    print(f"📁 Current directory: {current_dir}")
    print(f"📁 Looking for facial recognition in: {facial_recognition_path}")
    print(f"🔍 Facial recognition available: {FACIAL_RECOGNITION_AVAILABLE}")
    print("🌐 API will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")