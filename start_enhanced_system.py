#!/usr/bin/env python3
"""
Enhanced Security System Launcher with Demographics
Run this script to start the complete system with demographic analysis
"""

import subprocess
import time
import sys
import os
import signal

processes = []

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n🛑 Shutting down Security System...")
    for p in processes:
        try:
            p.terminate()
        except:
            pass
    sys.exit(0)

def check_dependencies():
    """Check if all required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'cv2',
        'numpy',
        'torch',
        'tensorflow',
        'deepface',
        'fastapi',
        'uvicorn',
        'psycopg2',
        'facenet_pytorch',
        'ultralytics',
        'easyocr'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n⚠️  Missing packages detected!")
        print("Install them with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    # Check DeepFace specifically
    try:
        from deepface import DeepFace
        print("   ✅ DeepFace - Demographics analysis available")
    except ImportError:
        print("   ⚠️  DeepFace not available - Demographics will be disabled")
        print("      Install with: pip install deepface")
    
    return True

def check_database():
    """Check if PostgreSQL is running and accessible"""
    print("\n🗄️  Checking database connection...")
    try:
        import psycopg2
        conn = psycopg2.connect(
            host='localhost',
            database='facial_recognition',
            user='postgres',
            password='incorect'
        )
        conn.close()
        print("   ✅ Database connection successful")
        return True
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")
        print("\n   Please ensure PostgreSQL is running and configured properly")
        return False

def start_backend():
    """Start the FastAPI backend with demographics"""
    print("\n🚀 Starting Backend Server (with Demographics)...")
    
    # Check if the enhanced app exists
    if not os.path.exists('app_with_demographics.py'):
        print("   ❌ app_with_demographics.py not found!")
        print("   Please ensure the enhanced backend file exists")
        return None
    
    process = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'app_with_demographics:app', 
         '--host', '0.0.0.0', '--port', '8000', '--log-level', 'info'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print("   ✅ Backend server starting on http://localhost:8000")
    print("   📚 API Docs: http://localhost:8000/docs")
    print("   🔌 WebSocket: ws://localhost:8000/ws")
    return process

def start_frontend():
    """Start the React frontend"""
    print("\n🎨 Starting Frontend Application...")
    
    # Check if frontend directory exists
    if not os.path.exists('frontend'):
        print("   ❌ frontend directory not found!")
        print("   Please ensure the React app is set up in ./frontend")
        return None
    
    # Change to frontend directory
    os.chdir('frontend')
    
    # Check if node_modules exists
    if not os.path.exists('node_modules'):
        print("   📦 Installing frontend dependencies...")
        subprocess.run(['npm', 'install'], check=True)
    
    # Start React app
    process = subprocess.Popen(
        ['npm', 'start'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Change back to original directory
    os.chdir('..')
    
    print("   ✅ Frontend starting on http://localhost:3000")
    return process

def main():
    """Main launcher function"""
    print("="*70)
    print("🔒 ENHANCED SECURITY SYSTEM WITH DEMOGRAPHICS")
    print("="*70)
    print("\nFeatures:")
    print("  ✅ Face Recognition")
    print("  ✅ License Plate Recognition")
    print("  ✅ Demographic Analysis (Age, Gender, Emotion)")
    print("  ✅ Multi-threaded Processing")
    print("  ✅ GPU Acceleration")
    print("="*70)
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again")
        sys.exit(1)
    
    # Check database
    if not check_database():
        print("\n❌ Please fix database connection and try again")
        sys.exit(1)
    
    print("\n🎬 Starting all services...")
    
    # Start backend
    backend = start_backend()
    if backend:
        processes.append(backend)
        time.sleep(3)  # Give backend time to start
    else:
        print("❌ Failed to start backend")
        sys.exit(1)
    
    # Start frontend
    frontend = start_frontend()
    if frontend:
        processes.append(frontend)
        time.sleep(3)  # Give frontend time to start
    else:
        print("❌ Failed to start frontend")
        for p in processes:
            p.terminate()
        sys.exit(1)
    
    print("\n" + "="*70)
    print("✅ SYSTEM READY!")
    print("="*70)
    print("\n📌 Access Points:")
    print("   🖥️  Frontend: http://localhost:3000")
    print("   🔌 Backend API: http://localhost:8000")
    print("   📚 API Docs: http://localhost:8000/docs")
    print("\n⚙️  Features:")
    print("   • Face Recognition with Demographics")
    print("   • License Plate Recognition")
    print("   • Real-time Video Streaming")
    print("   • Multi-threaded Processing")
    print("   • GPU Acceleration (if available)")
    print("\n🛑 Press Ctrl+C to stop all services")
    print("="*70)
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            for i, p in enumerate(processes):
                if p.poll() is not None:
                    print(f"\n⚠️  Process {i} has stopped unexpectedly!")
                    
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()