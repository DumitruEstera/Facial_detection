import torch
import numpy as np
import cv2
from typing import List, Optional
from facenet_pytorch import InceptionResnetV1


class FaceNetFeatureExtractor:
    """Face embedding extractor using facenet-pytorch's InceptionResnetV1 pretrained on VGGFace2."""

    def __init__(self, device: Optional[str] = None):
        # Decide whether to use GPU (CUDA) or fallback to CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Download (on first run) and load the pretrained FaceNet model
        # Facenet‑pytorch handles the caching automatically (<~90 MB)
        self.model = (
            InceptionResnetV1(pretrained="vggface2", classify=False)
            .eval()
            .to(self.device)
        )

        self.input_size = (160, 160)  # The network expects 160×160 aligned faces
        self.embedding_dim = 512  # InceptionResnetV1 outputs 512‑d vectors

    # ---------------------------------------------------------------------
    # Pre‑processing helpers
    # ---------------------------------------------------------------------
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """BGR (OpenCV) ⇒ normalized FloatTensor in [0,1] with shape (1, 3, 160, 160)."""
        # Convert BGR → RGB, resize, convert to float32 in [0,1]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, self.input_size)
        img_rgb = img_rgb.astype(np.float32) / 255.0

        # HWC → CHW, add batch dimension
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def preprocess_face(self, face_image: np.ndarray) -> torch.Tensor:
        """Alias kept for backward compatibility."""
        return self._to_tensor(face_image)

    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Return a single 512‑D L2‑normalised embedding for the given face."""
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
        """Vectorised version for a list of faces."""
        if not face_images:
            return []

        batch = torch.cat([self._to_tensor(img) for img in face_images], dim=0)
        with torch.no_grad():
            embs = self.model(batch).cpu().numpy()
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        return [e.astype(np.float32) for e in embs]

    # ------------------------------------------------------------------
    # Utility functions – unchanged API so the rest of the project works
    # ------------------------------------------------------------------
    @staticmethod
    def compute_distance(
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "euclidean",
    ) -> float:
        if metric == "euclidean":
            return float(np.linalg.norm(embedding1 - embedding2))
        if metric == "cosine":
            return float(1 - np.dot(embedding1, embedding2))
        raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def is_same_person(
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        threshold: float = 1.00,  # Typical L2 threshold for 512‑D FaceNet
    ) -> bool:
        return FaceNetFeatureExtractor.compute_distance(embedding1, embedding2) < threshold
