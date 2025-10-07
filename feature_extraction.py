import torch
import numpy as np
import cv2
from typing import List, Optional
from facenet_pytorch import InceptionResnetV1


class FaceNetFeatureExtractor:
    """GPU-accelerated Face embedding extractor"""

    def __init__(self, device: Optional[str] = None):
        # Auto-detect GPU
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"🎮 FaceNet using device: {self.device}")
        if self.device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        self.model = (
            InceptionResnetV1(pretrained="vggface2", classify=False)
            .eval()
            .to(self.device)
        )

        self.input_size = (160, 160)
        self.embedding_dim = 512

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert BGR image to GPU tensor"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, self.input_size)
        img_rgb = img_rgb.astype(np.float32) / 255.0

        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def preprocess_face(self, face_image: np.ndarray) -> torch.Tensor:
        return self._to_tensor(face_image)

    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract single embedding (GPU-accelerated)"""
        try:
            inp = self._to_tensor(face_image)
            with torch.no_grad():
                emb = self.model(inp)[0].cpu().numpy()
            emb = emb / np.linalg.norm(emb)
            return emb.astype(np.float32)
        except Exception as exc:
            print(f"Error extracting embedding: {exc}")
            return None

    def extract_embeddings_batch(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """Batch processing on GPU"""
        if not face_images:
            return []

        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(face_images), batch_size):
            batch = face_images[i:i + batch_size]
            batch_tensor = torch.cat([self._to_tensor(img) for img in batch], dim=0)
            
            with torch.no_grad():
                embs = self.model(batch_tensor).cpu().numpy()
            
            embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
            all_embeddings.extend([e.astype(np.float32) for e in embs])
        
        return all_embeddings

    @staticmethod
    def compute_distance(embedding1: np.ndarray, embedding2: np.ndarray,
                        metric: str = "euclidean") -> float:
        if metric == "euclidean":
            return float(np.linalg.norm(embedding1 - embedding2))
        if metric == "cosine":
            return float(1 - np.dot(embedding1, embedding2))
        raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def is_same_person(embedding1: np.ndarray, embedding2: np.ndarray,
                      threshold: float = 1.00) -> bool:
        return FaceNetFeatureExtractor.compute_distance(embedding1, embedding2) < threshold