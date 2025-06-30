import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime
import threading
import queue

# Import our modules (assuming they're in the same directory)
from database_manager import DatabaseManager
from face_detection import YuNetFaceDetector
from feature_extraction import FaceNetFeatureExtractor
from faiss_index import FaissIndex

class FacialRecognitionSystem:
    def __init__(self, db_config: Dict, camera_id: str = "0"):
        """
        Initialize the facial recognition system
        
        Args:
            db_config: Database configuration dictionary
            camera_id: Camera identifier
        """
        self.camera_id = camera_id
        
        # Initialize components
        print("Initializing database connection...")
        self.db = DatabaseManager(**db_config)
        self.db.connect()
        
        print("Initializing face detector...")
        self.face_detector = YuNetFaceDetector()
        
        print("Initializing feature extractor...")
        self.feature_extractor = FaceNetFeatureExtractor()
        
        print("Initializing Faiss index...")
        self.faiss_index = FaissIndex()
        
        # Load existing embeddings from database
        self._load_embeddings_from_db()
        
        # Recognition parameters
        self.recognition_threshold = 0.6
        self.min_confidence = 0.8
        
        # Threading for performance
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.is_running = False
        
    def _load_embeddings_from_db(self):
        """Load all existing embeddings from database into Faiss index"""
        try:
            embeddings_data = self.db.get_all_embeddings()
            
            if embeddings_data:
                print(f"Loading {len(embeddings_data)} embeddings into index...")
                self.faiss_index.rebuild_index(embeddings_data)
                print("Embeddings loaded successfully")
            else:
                print("No existing embeddings found in database")
                
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            
    def register_person(self, name: str, employee_id: str, face_images: List[np.ndarray], 
                       department: str = None, authorized_zones: List[str] = None) -> bool:
        """
        Register a new person in the system
        
        Args:
            name: Person's name
            employee_id: Employee ID
            face_images: List of face images for the person
            department: Department name
            authorized_zones: List of authorized zones
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Add person to database
            person_id = self.db.add_person(name, employee_id, department, authorized_zones)
            
            # Extract embeddings from face images
            embeddings = []
            for face_image in face_images:
                embedding = self.feature_extractor.extract_embedding(face_image)
                if embedding is not None:
                    embeddings.append(embedding)
                    
            if not embeddings:
                print(f"Failed to extract embeddings for {name}")
                return False
                
            # Add embeddings to database and Faiss index
            for embedding in embeddings:
                self.db.add_face_embedding(person_id, embedding)
                self.faiss_index.add_embedding(embedding, person_id)
                
            print(f"Successfully registered {name} with {len(embeddings)} face embeddings")
            return True
            
        except Exception as e:
            print(f"Error registering person: {e}")
            return False
            
    def recognize_face(self, face_image: np.ndarray) -> Optional[Dict]:
        """
        Recognize a face in an image
        
        Args:
            face_image: Face image to recognize
            
        Returns:
            Dictionary with person information and confidence, or None if not recognized
        """
        # Extract embedding
        embedding = self.feature_extractor.extract_embedding(face_image)
        if embedding is None:
            return None
            
        # Search in Faiss index
        results = self.faiss_index.search(embedding, k=1, threshold=self.recognition_threshold)
        
        if not results:
            return None
            
        # Get best match
        person_id, distance = results[0]
        
        # Calculate confidence (convert distance to confidence score)
        confidence = 1.0 - (distance / self.recognition_threshold)
        
        # Get person information
        person_info = self.db.get_person_by_id(person_id)
        if person_info:
            return {
                'person_id': person_id,
                'name': person_info['name'],
                'employee_id': person_info['employee_id'],
                'department': person_info['department'],
                'confidence': confidence,
                'distance': distance
            }
            
        return None
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame for face detection and recognition
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Annotated frame and list of recognition results
        """
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        
        recognition_results = []
        labels = []
        confidences = []
        
        # Process each detected face
        for face_bbox in faces:
            # Extract face region
            face_image = self.face_detector.extract_face(frame, face_bbox)
            if face_image is None:
                labels.append("Unknown")
                confidences.append(0.0)
                continue
                
            # Recognize face
            result = self.recognize_face(face_image)
            
            if result and result['confidence'] >= self.min_confidence:
                labels.append(result['name'])
                confidences.append(result['confidence'])
                recognition_results.append({
                    **result,
                    'bbox': face_bbox,
                    'timestamp': datetime.now()
                })
                
                # Log access
                self.db.log_access(
                    result['person_id'], 
                    self.camera_id, 
                    result['confidence']
                )
            else:
                labels.append("Unknown")
                confidences.append(0.0)
                
        # Draw faces on frame
        annotated_frame = self.face_detector.draw_faces(frame, faces, labels, confidences)
        
        return annotated_frame, recognition_results
        
    def start_video_stream(self, source: int = 0):
        """
        Start processing video stream from camera
        
        Args:
            source: Video source (0 for default camera, or video file path)
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
            
        print("Starting video stream... Press 'q' to quit")
        
        # FPS calculation
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            annotated_frame, results = self.process_frame(frame)
            
            # Calculate FPS
            fps_frame_count += 1
            if fps_frame_count >= 30:
                fps_end_time = time.time()
                fps = fps_frame_count / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                fps_frame_count = 0
                
            # Display FPS
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display results
            cv2.imshow("Facial Recognition System", annotated_frame)
            
            # Print recognition results
            for result in results:
                print(f"Recognized: {result['name']} (ID: {result['employee_id']}) "
                      f"with confidence: {result['confidence']:.2f}")
                
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    def capture_faces_for_registration(self, source: int = 0, num_faces: int = 10) -> List[np.ndarray]:
        """
        Capture face images for registration
        
        Args:
            source: Video source
            num_faces: Number of face images to capture
            
        Returns:
            List of captured face images
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return []
            
        print(f"Capturing {num_faces} face images. Press 'c' to capture, 'q' to quit")
        
        captured_faces = []
        
        while len(captured_faces) < num_faces:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            
            # Draw faces
            display_frame = self.face_detector.draw_faces(frame, faces)
            
            # Show capture count
            cv2.putText(display_frame, f"Captured: {len(captured_faces)}/{num_faces}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Face Capture", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') and len(faces) == 1:
                # Capture face
                face_image = self.face_detector.extract_face(frame, faces[0])
                if face_image is not None:
                    captured_faces.append(face_image)
                    print(f"Captured face {len(captured_faces)}/{num_faces}")
                    
            elif key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        return captured_faces
        
    def cleanup(self):
        """Clean up resources"""
        self.db.disconnect()
        print("System cleaned up")