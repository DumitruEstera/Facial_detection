#!/usr/bin/env python3
"""
Example usage of the Facial Recognition System
"""

import sys
import argparse
from facial_recognition_system import FacialRecognitionSystem

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'facial_recognition_db',
    'user': 'postgres',
    'password': 'admin'  # Change this to your PostgreSQL password
}

def main():
    parser = argparse.ArgumentParser(description='Facial Recognition System')
    parser.add_argument('--mode', choices=['stream', 'register', 'test'], 
                       default='stream', help='Operation mode')
    parser.add_argument('--name', type=str, help='Person name for registration')
    parser.add_argument('--employee-id', type=str, help='Employee ID for registration')
    parser.add_argument('--department', type=str, help='Department for registration')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--video', type=str, help='Video file path (instead of camera)')
    
    args = parser.parse_args()
    
    # Initialize system
    system = FacialRecognitionSystem(DB_CONFIG, camera_id=str(args.camera))
    
    try:
        if args.mode == 'register':
            # Registration mode
            if not args.name or not args.employee_id:
                print("Error: --name and --employee-id are required for registration")
                sys.exit(1)
                
            print(f"Starting registration for {args.name} (ID: {args.employee_id})")
            
            # Capture face images
            source = args.video if args.video else args.camera
            face_images = system.capture_faces_for_registration(source, num_faces=10)
            
            if not face_images:
                print("No faces captured. Registration cancelled.")
                sys.exit(1)
                
            # Register person
            success = system.register_person(
                name=args.name,
                employee_id=args.employee_id,
                face_images=face_images,
                department=args.department
            )
            
            if success:
                print(f"Successfully registered {args.name}")
            else:
                print("Registration failed")
                
        elif args.mode == 'stream':
            # Live stream mode
            source = args.video if args.video else args.camera
            system.start_video_stream(source)
            
        elif args.mode == 'test':
            # Test mode - print system statistics
            stats = system.faiss_index.get_statistics()
            print("\nSystem Statistics:")
            print(f"Total embeddings: {stats['total_embeddings']}")
            print(f"Unique persons: {stats['unique_persons']}")
            print(f"Index type: {stats['index_type']}")
            print(f"Embedding dimension: {stats['dimension']}")
            
    except KeyboardInterrupt:
        print("\nStopping system...")
    finally:
        system.cleanup()
        
if __name__ == "__main__":
    main()


"""
Usage Examples:

1. Start live facial recognition:
   python example_usage.py --mode stream

2. Register a new person:
   python example_usage.py --mode register --name "John Doe" --employee-id "EMP001" --department "Engineering"

3. Use a video file instead of camera:
   python example_usage.py --mode stream --video path/to/video.mp4

4. Test system and show statistics:
   python example_usage.py --mode test

5. Use a different camera:
   python example_usage.py --mode stream --camera 1
"""