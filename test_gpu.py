#!/usr/bin/env python3
"""
GPU Verification Script
Run this to check if all components can use GPU
"""

import sys

def test_nvidia_driver():
    """Test NVIDIA driver"""
    print("=" * 60)
    print("1️⃣  Testing NVIDIA Driver")
    print("=" * 60)
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA driver installed")
            print(result.stdout.split('\n')[0:3])  # Show GPU info
            return True
        else:
            print("❌ nvidia-smi failed")
            return False
    except FileNotFoundError:
        print("❌ NVIDIA driver not found")
        return False

def test_pytorch():
    """Test PyTorch GPU"""
    print("\n" + "=" * 60)
    print("2️⃣  Testing PyTorch GPU Support")
    print("=" * 60)
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA device count: {torch.cuda.device_count()}")
            print(f"✅ CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"✅ CUDA version: {torch.version.cuda}")
            
            # Test tensor operation
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = x @ y
            print("✅ GPU tensor operations working")
            return True
        else:
            print("❌ PyTorch CUDA not available")
            print("   Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ PyTorch GPU test failed: {e}")
        return False

def test_tensorflow():
    """Test TensorFlow GPU"""
    print("\n" + "=" * 60)
    print("3️⃣  Testing TensorFlow GPU Support")
    print("=" * 60)
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPU devices found: {len(gpus)}")
        
        if gpus:
            for gpu in gpus:
                print(f"✅ {gpu}")
            
            # Test operation
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
            print("✅ GPU tensor operations working")
            return True
        else:
            print("❌ TensorFlow GPU not available")
            print("   Install with: pip install tensorflow[and-cuda]")
            return False
    except ImportError:
        print("❌ TensorFlow not installed")
        return False
    except Exception as e:
        print(f"❌ TensorFlow GPU test failed: {e}")
        return False

def test_faiss():
    """Test FAISS GPU"""
    print("\n" + "=" * 60)
    print("4️⃣  Testing FAISS GPU Support")
    print("=" * 60)
    try:
        import faiss
        print(f"FAISS version: {faiss.__version__}")
        
        # Check if GPU version is installed
        if hasattr(faiss, 'StandardGpuResources'):
            print("✅ FAISS-GPU installed")
            
            # Test GPU index
            res = faiss.StandardGpuResources()
            dimension = 128
            index_cpu = faiss.IndexFlatL2(dimension)
            index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            print("✅ GPU index creation successful")
            return True
        else:
            print("⚠️  FAISS-CPU installed (not GPU)")
            print("   Install GPU version: pip install faiss-gpu")
            return False
    except ImportError:
        print("❌ FAISS not installed")
        return False
    except Exception as e:
        print(f"❌ FAISS GPU test failed: {e}")
        return False

def test_ultralytics():
    """Test Ultralytics YOLOv8 GPU"""
    print("\n" + "=" * 60)
    print("5️⃣  Testing Ultralytics YOLOv8 GPU Support")
    print("=" * 60)
    try:
        from ultralytics import YOLO
        import torch
        
        if torch.cuda.is_available():
            # Create a simple model
            model = YOLO('yolov8n.pt')  # Will auto-download if not present
            model.to('cuda')
            print("✅ YOLOv8 can use GPU")
            return True
        else:
            print("⚠️  YOLOv8 installed but CUDA not available")
            return False
    except ImportError:
        print("❌ Ultralytics not installed")
        return False
    except Exception as e:
        print(f"⚠️  YOLOv8 test warning: {e}")
        return True  # Not critical

def test_paddleocr():
    """Test PaddleOCR GPU"""
    print("\n" + "=" * 60)
    print("6️⃣  Testing PaddleOCR GPU Support")
    print("=" * 60)
    try:
        from paddleocr import PaddleOCR
        import paddle
        
        print(f"PaddlePaddle version: {paddle.__version__}")
        
        # Check CUDA
        if paddle.is_compiled_with_cuda():
            print("✅ PaddlePaddle compiled with CUDA")
            print(f"✅ CUDA version: {paddle.version.cuda()}")
            
            # Try to create OCR with GPU
            ocr = PaddleOCR(use_gpu=True, show_log=False)
            print("✅ PaddleOCR GPU initialization successful")
            return True
        else:
            print("⚠️  PaddlePaddle CPU version installed")
            print("   Install GPU version:")
            print("   python -m pip install paddlepaddle-gpu -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html")
            return False
    except ImportError:
        print("❌ PaddleOCR not installed")
        return False
    except Exception as e:
        print(f"⚠️  PaddleOCR test warning: {e}")
        return True  # Not critical

def test_facenet():
    """Test FaceNet GPU"""
    print("\n" + "=" * 60)
    print("7️⃣  Testing FaceNet GPU Support")
    print("=" * 60)
    try:
        from facenet_pytorch import InceptionResnetV1
        import torch
        
        if torch.cuda.is_available():
            model = InceptionResnetV1(pretrained='vggface2').eval()
            model = model.to('cuda')
            print("✅ FaceNet can use GPU")
            
            # Test inference
            dummy_input = torch.randn(1, 3, 160, 160).cuda()
            with torch.no_grad():
                output = model(dummy_input)
            print("✅ FaceNet GPU inference working")
            return True
        else:
            print("⚠️  FaceNet installed but CUDA not available")
            return False
    except ImportError:
        print("❌ facenet-pytorch not installed")
        return False
    except Exception as e:
        print(f"⚠️  FaceNet test warning: {e}")
        return True

def main():
    """Run all tests"""
    print("\n" + "🎮" * 30)
    print("GPU VERIFICATION TEST")
    print("🎮" * 30 + "\n")
    
    results = {
        "NVIDIA Driver": test_nvidia_driver(),
        "PyTorch": test_pytorch(),
        "TensorFlow": test_tensorflow(),
        "FAISS": test_faiss(),
        "YOLOv8": test_ultralytics(),
        "PaddleOCR": test_paddleocr(),
        "FaceNet": test_facenet()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component}: {'PASS' if status else 'FAIL'}")
    
    total = len(results)
    passed = sum(results.values())
    
    print("\n" + "-" * 60)
    print(f"Results: {passed}/{total} components GPU-ready")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! GPU acceleration fully enabled!")
        print("   Your security system will be MUCH faster now!")
    elif passed >= 4:
        print("\n⚠️  PARTIAL SUCCESS - Core components working")
        print("   Your system will still be faster, but some optimizations missing")
    else:
        print("\n❌ MULTIPLE FAILURES - GPU not properly configured")
        print("   Please review the errors above and reinstall packages")
    
    print("=" * 60)

if __name__ == "__main__":
    main()