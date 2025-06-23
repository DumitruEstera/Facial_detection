"""
API-friendly version of the facial recognition system
Modified to work with web interface instead of console input
"""

import cv2
import numpy as np
import os
import pickle
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import urllib.request

class APIFacialRecognitionSystem:
    def __init__(self):
        # Initialize YUNET face detector
        self.face_detector = None
        self.face_recognizer = None
        
        # Download and initialize YUNET model
        self.initialize_yunet()
        
        # Initialize face recognizer
        self.initialize_face_recognizer()
        
        # Face database
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_this_frame = True
        
        # Load known faces database
        self.load_known_faces()
        
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
        
        print("API Facial Recognition System initialized successfully!")
    
    def download_yunet_model(self):
        """Download YUNET model if not exists"""
        model_path = "face_detection_yunet_2023mar.onnx"
        if not os.path.exists(model_path):
            print("Downloading YUNET model...")
            
            # List of possible URLs to try
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
                    score_threshold=0.7,
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
    
    def extract_face_features(self, face_image):
        """Extract features from face image using histogram of oriented gradients"""
        # Resize face to standard size
        face_resized = cv2.resize(face_image, (64, 64))
        
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate HOG features
        hog = cv2.HOGDescriptor(
            _winSize=(64, 64),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
        
        features = hog.compute(gray_face)
        return features.flatten() if features is not None else np.array([])
    
    def preprocess_face_for_recognition(self, face_region, target_size=(100, 100)):
        """Standardize face preprocessing for both registration and recognition"""
        if face_region.size == 0:
            return None
        
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        resized_face = cv2.resize(gray_face, target_size)
        
        # Apply histogram equalization for better contrast
        equalized_face = cv2.equalizeHist(resized_face)
        
        return equalized_face
    
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
                        # Extract bounding box coordinates and convert to regular Python integers
                        x, y, w, h = [int(coord) for coord in face[:4]]
                        confidence = float(face[14])  # Face detection confidence
                        
                        # Ensure coordinates are within image bounds
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, width - x)
                        h = min(h, height - y)
                        
                        if w > 0 and h > 0:
                            # Convert landmarks to regular Python floats
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
                    # Convert to regular Python integers
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
    
    def register_new_face_api(self, frame, face_bbox, name):
        """API-friendly version of register_new_face - takes name as parameter"""
        if not name or not name.strip():
            return False, "Name cannot be empty"
        
        try:
            # Extract face region with consistent preprocessing
            x, y, w, h = face_bbox
            
            # Add some padding around the face
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(w + 2*padding, frame.shape[1] - x)
            h = min(h + 2*padding, frame.shape[0] - y)
            
            face_region = frame[y:y+h, x:x+w]
            
            if face_region.size > 0:
                if self.face_recognizer is not None:
                    # Use consistent preprocessing for LBPH
                    processed_face = self.preprocess_face_for_recognition(face_region)
                    if processed_face is not None:
                        encoding = processed_face  # Store the preprocessed face for LBPH
                    else:
                        return False, "Could not preprocess face"
                else:
                    # Use manual feature extraction
                    encoding = self.extract_face_features(face_region)
                
                if encoding is not None and len(encoding) > 0:
                    # Add to known faces
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(name.strip())
                    
                    # If using OpenCV recognizer, retrain it
                    if self.face_recognizer is not None:
                        self.retrain_recognizer()
                    
                    return True, f"Successfully registered face for: {name}"
                else:
                    return False, "Could not create face encoding"
            else:
                return False, "Invalid face region"
                
        except Exception as e:
            return False, f"Error registering face: {str(e)}"
    
    def load_known_faces(self):
        """Load known faces from database file"""
        try:
            if os.path.exists('yunet_face_database.pkl'):
                with open('yunet_face_database.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"Loaded {len(self.known_face_names)} known faces from database")
                
                # Retrain the recognizer with loaded data
                if self.face_recognizer is not None and len(self.known_face_encodings) > 0:
                    self.retrain_recognizer()
                    print("Face recognizer retrained with loaded data!")
                    
            else:
                print("No existing face database found. Starting with empty database.")
        except Exception as e:
            print(f"Error loading face database: {e}")
            self.known_face_encodings = []
            self.known_face_names = []
    
    def save_known_faces(self):
        """Save known faces to database file"""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }
            with open('yunet_face_database.pkl', 'wb') as f:
                pickle.dump(data, f)
            print("Face database saved successfully!")
            return True
        except Exception as e:
            print(f"Error saving face database: {e}")
            return False
    
    def retrain_recognizer(self):
        """Retrain the OpenCV face recognizer with all known faces"""
        if self.face_recognizer is not None and len(self.known_face_encodings) > 0:
            try:
                faces = []
                labels = []
                
                for i, face_data in enumerate(self.known_face_encodings):
                    if isinstance(face_data, np.ndarray) and face_data.ndim == 2:
                        faces.append(face_data)
                        labels.append(i)
                
                if len(faces) > 0:
                    self.face_recognizer.train(faces, np.array(labels))
                    print(f"Face recognizer retrained with {len(faces)} faces!")
                else:
                    print("No valid face data found for training!")
            except Exception as e:
                print(f"Error retraining recognizer: {e}")
    
    def recognize_faces(self, frame):
        """Recognize faces in the frame"""
        # Detect faces
        detected_faces = self.detect_faces(frame)
        recognized_faces = []
        
        for face_data in detected_faces:
            bbox = face_data['bbox']
            confidence = face_data['confidence']
            
            # Extract face region with consistent preprocessing
            x, y, w, h = bbox
            padding = 20
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(w + 2*padding, frame.shape[1] - x_pad)
            h_pad = min(h + 2*padding, frame.shape[0] - y_pad)
            
            face_region = frame[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
            
            name = "Unknown"
            match_confidence = 0.0
            
            if face_region.size > 0 and len(self.known_face_encodings) > 0:
                try:
                    if self.face_recognizer is not None:
                        # Use consistent preprocessing for recognition
                        processed_face = self.preprocess_face_for_recognition(face_region)
                        
                        if processed_face is not None:
                            label, confidence_score = self.face_recognizer.predict(processed_face)
                            
                            # Lower confidence score means better match for LBPH
                            if confidence_score < 80:  # Threshold for recognition
                                if label < len(self.known_face_names):
                                    name = self.known_face_names[label]
                                    match_confidence = max(0, 1.0 - (confidence_score / 100.0))
                    else:
                        # Use manual feature extraction
                        current_encoding = self.extract_face_features(face_region)
                        
                        if len(current_encoding) > 0:
                            # Compare with known faces
                            best_match_confidence = 0.0
                            best_match_name = "Unknown"
                            
                            for i, known_encoding in enumerate(self.known_face_encodings):
                                if isinstance(known_encoding, np.ndarray) and len(known_encoding) == len(current_encoding):
                                    similarity = cosine_similarity([current_encoding], [known_encoding])[0][0]
                                    
                                    if similarity > 0.7 and similarity > best_match_confidence:
                                        best_match_confidence = similarity
                                        best_match_name = self.known_face_names[i]
                            
                            if best_match_confidence > 0:
                                name = best_match_name
                                match_confidence = best_match_confidence
                except Exception as e:
                    print(f"Error in face recognition: {e}")
            
            # Ensure all data is JSON serializable
            bbox_list = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            landmarks_list = []
            
            if isinstance(face_data['landmarks'], np.ndarray):
                landmarks_list = [[float(point[0]), float(point[1])] for point in face_data['landmarks']]
            else:
                landmarks_list = face_data['landmarks']
            
            recognized_faces.append({
                'bbox': bbox_list,
                'name': str(name),
                'confidence': float(match_confidence),
                'detection_confidence': float(confidence),
                'landmarks': landmarks_list
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
    
    def cleanup(self):
        """Clean up resources"""
        if self.camera_available and self.video_capture:
            self.video_capture.release()
        print("Facial recognition system cleaned up")