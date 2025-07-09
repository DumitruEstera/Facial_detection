import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime
import queue

from database_manager import DatabaseManager
from face_detection import YuNetFaceDetector
from feature_extraction import FaceNetFeatureExtractor
from faiss_index import FaissIndex


class FacialRecognitionSystem:
    def __init__(self, db_config: Dict, camera_id: str = "0"):
        """
        Initialize the facial recognition system
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
        self.faiss_index = FaissIndex(dimension=self.feature_extractor.embedding_dim)

        # Load existing embeddings from database
        self._load_embeddings_from_db()

        # Recognition parameters
        self.recognition_threshold = 1.4  # L2 distance threshold
        self.min_confidence = 0.6

        # Queues for multi‑threaded processing
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.is_running = False

    # --------------------------------------------------------------------- #
    #                         INITIALIZATION HELPERS                        #
    # --------------------------------------------------------------------- #

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

    # --------------------------------------------------------------------- #
    #                           REGISTRATION API                            #
    # --------------------------------------------------------------------- #

    def register_person(self,
                        name: str,
                        employee_id: str,
                        face_images: List[np.ndarray],
                        department: str = None,
                        authorized_zones: List[str] = None) -> bool:
        """
        Register a new person in the system.
        """
        try:
            # Add person to database
            person_id = self.db.add_person(name, employee_id, department, authorized_zones)

            embeddings = []
            for face_image in face_images:
                embedding = self.feature_extractor.extract_embedding(face_image)
                if embedding is not None:
                    embeddings.append(embedding)

            if not embeddings:
                print(f"Failed to extract embeddings for {name}")
                return False

            # Persist embeddings
            for embedding in embeddings:
                self.db.add_face_embedding(person_id, embedding)
                self.faiss_index.add_embedding(embedding, person_id)

            print(f"Successfully registered {name} with {len(embeddings)} face embeddings")
            return True

        except Exception as e:
            print(f"Error registering person: {e}")
            return False

    # --------------------------------------------------------------------- #
    #                       NEW: LOAD FACES FROM FILES                      #
    # --------------------------------------------------------------------- #

    def load_faces_from_files(self,
                              image_paths: List[str],
                              num_faces: int = 20) -> List[np.ndarray]:
        """
        Load and validate face images from disk.

        * Only keeps images that contain **exactly one** detectable face.
        * Stops when `num_faces` valid images have been collected.

        Args:
            image_paths: List of image file paths.
            num_faces: Target number of valid face crops to return.

        Returns:
            List of cropped face images.
        """
        valid_faces: List[np.ndarray] = []

        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"[WARN] Could not read image: {path}")
                continue

            faces = self.face_detector.detect_faces(img)
            if len(faces) != 1:
                print(f"[WARN] Skipping {path}: expected 1 face, found {len(faces)}")
                continue

            face_img = self.face_detector.extract_face(img, faces[0])
            if face_img is None:
                print(f"[WARN] Could not extract face from {path}")
                continue

            # Sanity check: can we embed?
            if self.feature_extractor.extract_embedding(face_img) is None:
                print(f"[WARN] Could not extract embedding from {path}")
                continue

            valid_faces.append(face_img)
            print(f"[INFO] Added face from {path} ({len(valid_faces)}/{num_faces})")

            if len(valid_faces) >= num_faces:
                break

        return valid_faces

    # --------------------------------------------------------------------- #
    #                       RECOGNITION & STREAMING                         #
    # --------------------------------------------------------------------- #

    def recognize_face(self, face_image: np.ndarray) -> Optional[Dict]:
        """
        Recognize a face in an image and return result dict or None.
        """
        embedding = self.feature_extractor.extract_embedding(face_image)
        if embedding is None:
            return None

        results = self.faiss_index.search(embedding, k=1, threshold=self.recognition_threshold)
        if not results:
            return None

        person_id, distance = results[0]
        confidence = 1.0 - (distance / self.recognition_threshold)
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
        Detect and recognize faces in a frame; return annotated frame plus results.
        """
        faces = self.face_detector.detect_faces(frame)
        recognition_results = []
        labels, confidences = [], []

        for face_bbox in faces:
            face_image = self.face_detector.extract_face(frame, face_bbox)
            if face_image is None:
                labels.append("Unknown")
                confidences.append(0.0)
                continue

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
                self.db.log_access(result['person_id'], self.camera_id, result['confidence'])
            else:
                labels.append("Unknown")
                confidences.append(0.0)

        annotated_frame = self.face_detector.draw_faces(frame, faces, labels, confidences)
        return annotated_frame, recognition_results

    def start_video_stream(self, source: int = 0):
        """
        Start processing a live camera or video file stream.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Could not open video source")
            return

        print("Starting video stream... (press 'q' to quit)")

        # FPS calculation
        fps_start_time = time.time()
        fps_frame_count, fps = 0, 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, results = self.process_frame(frame)

            # FPS
            fps_frame_count += 1
            if fps_frame_count >= 30:
                fps_end_time = time.time()
                fps = fps_frame_count / (fps_end_time - fps_start_time)
                fps_start_time, fps_frame_count = fps_end_time, 0

            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Facial Recognition System", annotated_frame)

            for result in results:
                print(f"Recognized: {result['name']} (ID: {result['employee_id']}) "
                      f"with confidence: {result['confidence']:.2f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # --------------------------------------------------------------------- #
    #                   CAPTURE FACES FROM CAMERA (UPDATED)                 #
    # --------------------------------------------------------------------- #

    def capture_faces_for_registration(self,
                                       source: int = 0,
                                       num_faces: int = 20) -> List[np.ndarray]:
        """
        Capture face images for registration via live camera or video file.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Could not open video source")
            return []

        print(f"Capturing {num_faces} face images. Press 'c' to capture, 'q' to quit")

        captured_faces: List[np.ndarray] = []

        while len(captured_faces) < num_faces:
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.face_detector.detect_faces(frame)
            display_frame = self.face_detector.draw_faces(frame, faces)

            cv2.putText(display_frame,
                        f"Captured: {len(captured_faces)}/{num_faces}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Face Capture", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and len(faces) == 1:
                face_img = self.face_detector.extract_face(frame, faces[0])
                if face_img is not None:
                    captured_faces.append(face_img)
                    print(f"Captured face {len(captured_faces)}/{num_faces}")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return captured_faces

    # --------------------------------------------------------------------- #
    #                               CLEANUP                                 #
    # --------------------------------------------------------------------- #

    def cleanup(self):
        """Disconnect database and release resources."""
        self.db.disconnect()
        print("System cleaned up")
