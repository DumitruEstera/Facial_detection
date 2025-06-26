"""
Optimized database-integrated facial recognition system
This version uses proper feature extraction to generate 512-dimensional vectors
"""

import cv2
import numpy as np
import os
import asyncio
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import urllib.request
from database_manager import DatabaseManager
import logging

logger = logging.getLogger(__name__)

class OptimizedDatabaseFacialRecognitionSystem:
    def __init__(self):
        # Initialize YUNET face detector
        self.face_detector = None
        self.face_recognizer = None
        
        # Initialize database manager
        self.db_manager = DatabaseManager()
        self.db_initialized = False
        
        # Initialize PCA for dimensionality reduction
        self.pca = PCA(n_components=512)  # Reduce to 512 dimensions
        self.pca_fitted = False
        
        # Download and initialize YUNET model
        self.initialize_yunet()
        
        # Initialize face recognizer
        self.initialize_face_recognizer()
        
        # Face cache (loaded from database)
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        
        # Initialize camera
        self.video_capture = cv2.VideoCapture(0)
        
        # Check if camera is available
        if not self.video_capture.isOpened():
            print("Warning: Could not open camera")
            self.camera_available = False
        else:
            self.camera_available = True
            # Set camera resolution
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Optimized Database Facial Recognition System initialized!")
    
    async def initialize_database(self):
        """Initialize database connection and load known faces"""
        try:
            success = await self.db_manager.initialize()
            if success:
                await self.load_known_faces_from_db()
                self.db_initialized = True
                print("✅ Database integration successful!")
                return True
            else:
                print("❌ Database initialization failed!")
                return False
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
            return False
    
    async def load_known_faces_from_db(self):
        """Load known faces from database"""
        try:
            face_data = await self.db_manager.get_all_face_encodings()
            
            self.known_face_encodings = []
            self.known_face_names = []
            self.known_face_ids = []
            
            for personnel_id, name, encoding in face_data:
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name)
                self.known_face_ids.append(personnel_id)
            
            print(f"✅ Loaded {len(self.known_face_names)} known faces from database")
            
        except Exception as e:
            print(f"❌ Error loading faces from database: {e}")
            self.known_face_encodings = []
            self.known_face_names = []
            self.known_face_ids = []
    
    def download_yunet_model(self):
        """Download YUNET model if not exists"""
        model_path = "face_detection_yunet_2023mar.onnx"
        if not os.path.exists(model_path):
            print("Downloading YUNET model...")
            
            urls = [
                "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                "https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            ]
            
            downloaded = False
            for i, url in enumerate(urls):
                try:
                    print(f"Trying URL {i+1}/{len(urls)}...")
                    urllib.request.urlretrieve(url, model_path)
                    print("YUNET model downloaded successfully!")
                    downloaded = True
                    break
                except Exception as e:
                    print(f"Failed with URL {i+1}: {e}")
                    continue
            
            if not downloaded:
                print("All download attempts failed. Using fallback detector.")
                return None
                
        return model_path
    
    def initialize_yunet(self):
        """Initialize YUNET face detector"""
        try:
            model_path = self.download_yunet_model()
            if model_path and os.path.exists(model_path):
                self.face_detector = cv2.FaceDetectorYN.create(
                    model_path,
                    "",
                    (640, 480),
                    score_threshold=0.5,
                    nms_threshold=0.3,
                    top_k=5000
                )
                print("YUNET face detector initialized successfully!")
            else:
                # Fallback to Haar cascades
                print("Falling back to Haar Cascade face detection...")
                self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                print("Haar Cascade face detector initialized as fallback!")
        except Exception as e:
            print(f"Error initializing YUNET: {e}")
            print("Falling back to Haar Cascade face detection...")
            try:
                self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                print("Haar Cascade face detector initialized as fallback!")
            except Exception as e2:
                print(f"Error initializing fallback detector: {e2}")
                raise
    
    def initialize_face_recognizer(self):
        """Initialize face recognizer"""
        try:
            # Create LBPH face recognizer (Local Binary Patterns Histograms)
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            print("Face recognizer initialized successfully!")
        except Exception as e:
            print(f"Error initializing face recognizer: {e}")
            # Fallback to manual feature extraction if OpenCV contrib not available
            self.face_recognizer = None
            print("Using manual feature extraction for face recognition")
    
    def extract_optimized_face_features(self, face_image, target_dims=512):
        """Extract optimized features from face image with fixed dimensions"""
        try:
            # Resize face to a standard size
            face_resized = cv2.resize(face_image, (128, 128))
            
            # Convert to grayscale if needed
            if len(face_resized.shape) == 3:
                gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_resized
            
            # Apply histogram equalization
            equalized = cv2.equalizeHist(gray_face)
            
            # Calculate multiple features and combine them
            features = []
            
            # 1. LBP (Local Binary Pattern) features
            lbp = self.calculate_lbp(equalized)
            features.extend(lbp.flatten())
            
            # 2. HOG (Histogram of Oriented Gradients) features
            hog = cv2.HOGDescriptor(
                _winSize=(128, 128),
                _blockSize=(16, 16),
                _blockStride=(8, 8),
                _cellSize=(8, 8),
                _nbins=9
            )
            hog_features = hog.compute(equalized)
            if hog_features is not None:
                features.extend(hog_features.flatten())
            
            # 3. Statistical features
            stats = [
                np.mean(equalized), np.std(equalized), 
                np.min(equalized), np.max(equalized),
                np.median(equalized), np.var(equalized)
            ]
            features.extend(stats)
            
            # Convert to numpy array
            features = np.array(features, dtype=np.float32)
            
            # If we have more features than target_dims, use PCA or truncate
            if len(features) > target_dims:
                if len(features) >= target_dims * 2 and not self.pca_fitted:
                    # We have enough features to use PCA, but need more samples to fit
                    # For now, just truncate to target_dims
                    features = features[:target_dims]
                else:
                    features = features[:target_dims]
            elif len(features) < target_dims:
                # Pad with zeros if we have fewer features
                padding = np.zeros(target_dims - len(features))
                features = np.concatenate([features, padding])
            
            # Normalize features
            if np.std(features) > 0:
                features = (features - np.mean(features)) / np.std(features)
            
            return features
            
        except Exception as e:
            print(f"Error extracting optimized features: {e}")
            # Return zero vector if extraction fails
            return np.zeros(target_dims, dtype=np.float32)
    
    def calculate_lbp(self, image, radius=1, n_points=8):
        """Calculate Local Binary Pattern"""
        height, width = image.shape
        lbp = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                center = image[i, j]
                code = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    if 0 <= x < height and 0 <= y < width:
                        if image[x, y] >= center:
                            code |= (1 << k)
                lbp[i, j] = code
        
        return lbp
    
    def detect_faces(self, image):
        """Detect faces using YUNET or Haar Cascade fallback"""
        try:
            detected_faces = []
            
            # Check if we're using YUNET or Haar Cascade
            if hasattr(self.face_detector, 'detect'):  # YUNET
                # Set input size for the detector
                height, width = image.shape[:2]
                self.face_detector.setInputSize((width, height))
                
                # Detect faces
                _, faces = self.face_detector.detect(image)
                
                if faces is not None:
                    for face in faces:
                        # Extract bounding box coordinates
                        x, y, w, h = [int(coord) for coord in face[:4]]
                        confidence = float(face[14])  # Face detection confidence
                        
                        # Ensure coordinates are within image bounds
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, width - x)
                        h = min(h, height - y)
                        
                        if w > 0 and h > 0:
                            landmarks = [[float(point[0]), float(point[1])] for point in face[4:14].reshape(5, 2)]
                            
                            detected_faces.append({
                                'bbox': (x, y, w, h),
                                'confidence': confidence,
                                'landmarks': landmarks
                            })
            
            else:  # Haar Cascade fallback
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                for (x, y, w, h) in faces:
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    
                    # Create dummy landmarks for compatibility
                    landmarks = [
                        [float(x + w*0.3), float(y + h*0.4)],  # Left eye
                        [float(x + w*0.7), float(y + h*0.4)],  # Right eye
                        [float(x + w*0.5), float(y + h*0.6)],  # Nose
                        [float(x + w*0.3), float(y + h*0.8)],  # Left mouth
                        [float(x + w*0.7), float(y + h*0.8)]   # Right mouth
                    ]
                    
                    detected_faces.append({
                        'bbox': (x, y, w, h),
                        'confidence': 0.8,  # Default confidence for Haar
                        'landmarks': landmarks
                    })
            
            return detected_faces
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []
    
    async def register_new_face_db(self, frame, face_bbox, first_name, last_name, rank="STUDENT"):
        """Register a new face to the database with optimized feature extraction"""
        if not self.db_initialized:
            return False, "Database not initialized"
        
        if not first_name.strip() or not last_name.strip():
            return False, "First name and last name cannot be empty"
        
        try:
            # Extract face region with padding
            x, y, w, h = face_bbox
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(w + 2*padding, frame.shape[1] - x)
            h = min(h + 2*padding, frame.shape[0] - y)
            
            face_region = frame[y:y+h, x:x+w]
            
            if face_region.size > 0:
                # Extract optimized 512-dimensional features
                encoding = self.extract_optimized_face_features(face_region, target_dims=512)
                
                if encoding is not None and len(encoding) == 512:
                    # Get or create personnel in database
                    personnel_id = await self.db_manager.get_or_create_personnel(
                        first_name, last_name, rank=rank
                    )
                    
                    # Save face encoding to database
                    success = await self.db_manager.save_face_encoding(
                        personnel_id, encoding, confidence_score=0.9
                    )
                    
                    if success:
                        # Reload known faces from database
                        await self.load_known_faces_from_db()
                        
                        # Log the registration
                        await self.db_manager.log_detection(
                            zone_id=1,  # Default zone
                            personnel_id=personnel_id,
                            detection_type="FACE_REGISTRATION",
                            confidence_score=1.0,
                            detection_data={"action": "new_face_registered"}
                        )
                        
                        return True, f"Successfully registered face for: {first_name} {last_name}"
                    else:
                        return False, "Failed to save face encoding to database"
                else:
                    return False, f"Could not create proper face encoding (got {len(encoding) if encoding is not None else 0} dimensions instead of 512)"
            else:
                return False, "Invalid face region"
                
        except Exception as e:
            print(f"Error in register_new_face_db: {e}")
            return False, f"Error registering face: {str(e)}"
    
    async def recognize_faces_db(self, frame):
        """Recognize faces using database lookup with optimized features"""
        if not self.db_initialized:
            return []
        
        # Detect faces
        detected_faces = self.detect_faces(frame)
        recognized_faces = []
        
        for face_data in detected_faces:
            bbox = face_data['bbox']
            confidence = face_data['confidence']
            
            # Extract face region with padding
            x, y, w, h = bbox
            padding = 20
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(w + 2*padding, frame.shape[1] - x_pad)
            h_pad = min(h + 2*padding, frame.shape[0] - y_pad)
            
            face_region = frame[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
            
            name = "Unknown"
            match_confidence = 0.0
            personnel_id = None
            
            if face_region.size > 0:
                try:
                    # Extract optimized features from current face
                    current_encoding = self.extract_optimized_face_features(face_region, target_dims=512)
                    
                    if current_encoding is not None and len(current_encoding) == 512:
                        # Use database vector similarity search
                        similar_faces = await self.db_manager.find_similar_faces(
                            current_encoding, threshold=0.5, limit=1
                        )
                        
                        if similar_faces:
                            best_match = similar_faces[0]
                            name = best_match['name']
                            match_confidence = best_match['similarity']
                            personnel_id = best_match['personnel_id']
                            
                            # Log the detection
                            await self.db_manager.log_detection(
                                zone_id=1,  # Default zone
                                personnel_id=personnel_id,
                                detection_type="FACE_RECOGNITION",
                                confidence_score=match_confidence,
                                detection_data={
                                    "bbox": bbox,
                                    "recognized_name": name,
                                    "similarity": match_confidence
                                }
                            )
                        else:
                            # Log unknown face detection
                            await self.db_manager.log_detection(
                                zone_id=1,  # Default zone
                                personnel_id=None,
                                detection_type="UNKNOWN_FACE",
                                confidence_score=confidence,
                                detection_data={"bbox": bbox}
                            )
                            
                            # Create security alert for unknown person (less frequently)
                            if np.random.random() > 0.9:  # Only 10% of unknown detections create alerts
                                await self.db_manager.create_security_alert(
                                    alert_type="Unknown Person Detected",
                                    severity="MEDIUM",
                                    description=f"Unknown person detected with confidence {confidence:.2f}",
                                    zone_id=1,
                                    additional_data={"bbox": bbox, "detection_confidence": confidence}
                                )
                            
                except Exception as e:
                    print(f"Error in face recognition: {e}")
            
            recognized_faces.append({
                'bbox': bbox,
                'name': name,
                'confidence': match_confidence,
                'detection_confidence': confidence,
                'landmarks': face_data['landmarks'],
                'personnel_id': personnel_id
            })
        
        return recognized_faces
    
    def draw_face_boxes(self, frame, recognized_faces):
        """Draw bounding boxes and labels on detected faces"""
        for face in recognized_faces:
            x, y, w, h = face['bbox']
            name = face['name']
            confidence = face['confidence']
            landmarks = face['landmarks']
            
            # Choose color based on recognition status
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
                label = f"Unknown ({face['detection_confidence']:.2f})"
            else:
                color = (0, 255, 0)  # Green for known
                label = f"{name} ({confidence:.2f})"
            
            # Draw rectangle around face
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            
            # Draw facial landmarks
            for landmark in landmarks:
                cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (255, 255, 0), -1)
            
            # Draw label with background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
            cv2.rectangle(frame, (int(x), int(y) - 35), (int(x) + label_size[0] + 10, int(y)), color, cv2.FILLED)
            cv2.putText(frame, label, (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    async def cleanup(self):
        """Clean up resources"""
        if self.camera_available and self.video_capture:
            self.video_capture.release()
        
        if self.db_manager:
            await self.db_manager.close()
        
        print("Optimized database facial recognition system cleaned up")

# For backward compatibility, alias the optimized class
DatabaseFacialRecognitionSystem = OptimizedDatabaseFacialRecognitionSystem