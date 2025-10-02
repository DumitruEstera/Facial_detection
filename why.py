# test_imports.py
print("Testing imports...")

try:
    import torch
    print(f"✅ PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"❌ PyTorch error: {e}")

try:
    from facenet_pytorch import InceptionResnetV1
    print("✅ facenet-pytorch imported successfully")
    model = InceptionResnetV1(pretrained='vggface2')
    print("✅ FaceNet model loaded successfully")
except Exception as e:
    print(f"❌ facenet-pytorch error: {e}")

try:
    from ultralytics import YOLO
    print("✅ Ultralytics imported successfully")
except Exception as e:
    print(f"❌ Ultralytics error: {e}")

try:
    from paddleocr import PaddleOCR
    print("✅ PaddleOCR imported successfully")
except Exception as e:
    print(f"❌ PaddleOCR error: {e}")

print("\n Testing feature extraction...")
try:
    from feature_extraction import FaceNetFeatureExtractor
    extractor = FaceNetFeatureExtractor()
    print("✅ Feature extractor initialized successfully")
except Exception as e:
    print(f"❌ Feature extraction error: {e}")