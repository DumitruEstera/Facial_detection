#!/usr/bin/env python3
"""
Example usage of the License Plate Recognition System
"""

import sys
import argparse
from license_plate_recognition_system import LicensePlateRecognitionSystem
from database_manager import DatabaseManager

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'facial_recognition_db',  # Same database as facial recognition
    'user': 'postgres',
    'password': 'admin'  # Change this to your PostgreSQL password
}

def main():
    parser = argparse.ArgumentParser(description='License Plate Recognition System')
    parser.add_argument('--mode', choices=['stream', 'register', 'test-image', 'stats'], 
                       default='stream', help='Operation mode')
    parser.add_argument('--plate', type=str, help='License plate number for registration')
    parser.add_argument('--owner', type=str, help='Owner name for registration')
    parser.add_argument('--owner-id', type=str, help='Owner ID/Employee ID')
    parser.add_argument('--vehicle-type', type=str, help='Vehicle type (car, truck, etc.)')
    parser.add_argument('--authorized', type=bool, default=True, 
                       help='Whether the plate is authorized')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--video', type=str, help='Video file path (instead of camera)')
    parser.add_argument('--image', type=str, help='Image file path for testing')
    parser.add_argument('--save-images', action='store_true', 
                       help='Save captured vehicle images')
    
    args = parser.parse_args()
    
    db_manager = DatabaseManager(**DB_CONFIG)
    db_manager.connect()
    
    # Initialize system with the DatabaseManager object
    system = LicensePlateRecognitionSystem(db_manager)
    
    try:
        if args.mode == 'register':
            # Registration mode
            if not args.plate:
                print("Error: --plate is required for registration")
                sys.exit(1)
                
            print(f"Registering plate: {args.plate}")
            
            success = system.register_plate(
                plate_number=args.plate,
                vehicle_type=args.vehicle_type,
                owner_name=args.owner,
                owner_id=args.owner_id,
                is_authorized=args.authorized
            )
            
            if success:
                print(f"Successfully registered plate: {args.plate}")
            else:
                print("Registration failed")
                
        elif args.mode == 'stream':
            # Live stream mode
            source = args.video if args.video else args.camera
            system.start_video_stream(source)
            
        elif args.mode == 'test-image':
            # Test on single image
            if not args.image:
                print("Error: --image is required for test-image mode")
                sys.exit(1)
                
            system.test_on_image(args.image)
            
        elif args.mode == 'stats':
            # Show statistics
            stats = system.get_statistics()
            print("\nSystem Statistics:")
            print(f"Total license plates: {stats['total_plates']}")
            print(f"Authorized plates: {stats['authorized_plates']}")
            print(f"Total vehicle accesses: {stats['total_vehicle_accesses']}")
            print(f"Unauthorized accesses: {stats['unauthorized_vehicle_accesses']}")
            print(f"Cached authorized plates: {stats['cached_authorized_plates']}")
            
    except KeyboardInterrupt:
        print("\nStopping system...")
    finally:
        # Always close database connection
        db_manager.disconnect()
        
if __name__ == "__main__":
    main()


"""
Usage Examples:

1. Start live license plate recognition:
   python license_plate_example.py --mode stream --save-images

2. Register a new authorized plate:
   python license_plate_example.py --mode register --plate "B 123 ABC" --owner "John Doe" --owner-id "EMP001" --vehicle-type "car"

3. Register an unauthorized plate:
   python license_plate_example.py --mode register --plate "B 456 XYZ" --authorized False

4. Use a video file instead of camera:
   python license_plate_example.py --mode stream --video path/to/video.mp4

5. Test on a single image:
   python license_plate_example.py --mode test-image --image path/to/car_image.jpg

6. Show system statistics:
   python license_plate_example.py --mode stats

7. Use a different camera:
   python license_plate_example.py --mode stream --camera 1

Note: Make sure to install required packages:
pip install opencv-python numpy ultralytics easyocr torch psycopg2-binary
"""