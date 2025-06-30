import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from typing import List, Optional
import urllib.request
import zipfile
import os

class FaceNetFeatureExtractor:
    def __init__(self, model_path: str = None):
        """
        Initialize FaceNet feature extractor
        
        Args:
            model_path: Path to FaceNet model
        """
        if model_path is None:
            model_path = self._download_model()
            
        # Load FaceNet model
        self.model = keras.models.load_model(model_path)
        self.input_size = (160, 160)  # FaceNet input size
        
    def _download_model(self) -> str:
        """Download pre-trained FaceNet model"""
        # Note: In production, you should use a proper FaceNet model
        # This is a placeholder for the model loading logic
        model_dir = "facenet_model"
        
        if not os.path.exists(model_dir):
            print("Note: You need to download a pre-trained FaceNet model")
            print("You can use models from: https://github.com/davidsandberg/facenet")
            print("Or train your own using Keras/TensorFlow")
            os.makedirs(model_dir, exist_ok=True)
            
            # Create a simple placeholder model for demonstration
            # In production, replace this with actual FaceNet model
            inputs = keras.Input(shape=(160, 160, 3))
            x = keras.layers.Conv2D(32, 3, activation='relu')(inputs)
            x = keras.layers.MaxPooling2D(2)(x)
            x = keras.layers.Conv2D(64, 3, activation='relu')(x)
            x = keras.layers.MaxPooling2D(2)(x)
            x = keras.layers.Conv2D(128, 3, activation='relu')(x)
            x = keras.layers.GlobalAveragePooling2D()(x)
            x = keras.layers.Dense(256, activation='relu')(x)
            outputs = keras.layers.Dense(128)(x)  # 128-dimensional embedding
            
            model = keras.Model(inputs, outputs)
            model_path = os.path.join(model_dir, "facenet_placeholder.h5")
            model.save(model_path)
            
            print(f"Created placeholder model at {model_path}")
            print("Replace this with actual FaceNet model for production use")
            
        return os.path.join(model_dir, "facenet_placeholder.h5")
        
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for FaceNet
        
        Args:
            face_image: Face image in BGR format
            
        Returns:
            Preprocessed face image
        """
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Resize to FaceNet input size
        face_resized = cv2.resize(face_rgb, self.input_size)
        
        # Normalize pixel values
        face_normalized = face_resized.astype(np.float32)
        mean = np.array([127.5, 127.5, 127.5])
        face_normalized = (face_normalized - mean) / 128.0
        
        return face_normalized
        
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract feature embedding from a face image
        
        Args:
            face_image: Face image in BGR format
            
        Returns:
            128-dimensional face embedding or None if extraction fails
        """
        try:
            # Preprocess face
            face_processed = self.preprocess_face(face_image)
            
            # Add batch dimension
            face_batch = np.expand_dims(face_processed, axis=0)
            
            # Extract embedding
            embedding = self.model.predict(face_batch)[0]
            
            # L2 normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
            
    def extract_embeddings_batch(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract embeddings for multiple face images
        
        Args:
            face_images: List of face images in BGR format
            
        Returns:
            List of face embeddings
        """
        embeddings = []
        
        # Process faces in batches
        batch_size = 32
        for i in range(0, len(face_images), batch_size):
            batch_faces = face_images[i:i + batch_size]
            
            # Preprocess batch
            processed_batch = []
            for face in batch_faces:
                processed = self.preprocess_face(face)
                processed_batch.append(processed)
                
            # Convert to numpy array
            batch_array = np.array(processed_batch)
            
            # Extract embeddings
            batch_embeddings = self.model.predict(batch_array)
            
            # L2 normalize embeddings
            for embedding in batch_embeddings:
                normalized = embedding / np.linalg.norm(embedding)
                embeddings.append(normalized)
                
        return embeddings
        
    def compute_distance(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                        metric: str = 'euclidean') -> float:
        """
        Compute distance between two embeddings
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            metric: Distance metric ('euclidean' or 'cosine')
            
        Returns:
            Distance between embeddings
        """
        if metric == 'euclidean':
            return np.linalg.norm(embedding1 - embedding2)
        elif metric == 'cosine':
            return 1 - np.dot(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
    def is_same_person(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                      threshold: float = 0.6) -> bool:
        """
        Determine if two embeddings belong to the same person
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Distance threshold for same person
            
        Returns:
            True if same person, False otherwise
        """
        distance = self.compute_distance(embedding1, embedding2)
        return distance < threshold