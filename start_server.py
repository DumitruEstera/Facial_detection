#!/usr/bin/env python3
"""
Simple server startup script for the Security System API
This avoids the reload/import warnings
"""

import uvicorn
import sys
import os

def main():
    print("🔒 Starting Security System API Server...")
    print("=" * 50)
    print("📡 API will be available at: http://localhost:8000")
    print("🔌 WebSocket endpoint: ws://localhost:8000/ws")
    print("📚 API documentation: http://localhost:8000/docs")
    print("🎛️  Admin interface: http://localhost:8000/redoc")
    print("=" * 50)
    print("🔧 Make sure your database is running (PostgreSQL)")
    print("📹 Connect your camera and test it first")
    print("=" * 50)
    
    try:
        # Start the server
        uvicorn.run(
            "app:app", 
            host="0.0.0.0", 
            port=8000,
            reload=False,  # Disable reload to avoid warnings
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down API server... Goodbye!")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure no other service is using port 8000")
        print("2. Check if all dependencies are installed: pip install -r requirements.txt")
        print("3. Verify database connection settings in app.py")
        print("4. Try running manually: uvicorn app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

if __name__ == "__main__":
    main()