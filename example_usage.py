#!/usr/bin/env python3
"""
Example usage of the Facial Recognition System

Changes for 2025-07-09:
* Added ability to register a person from existing images instead of the camera.
* Increased default number of required face images from 10 to 20.
"""

import sys
import argparse
import os
import glob

from facial_recognition_system import FacialRecognitionSystem

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'facial_recognition_db',
    'user': 'postgres',
    'password': 'admin'  # Change this to your PostgreSQL password
}

DEFAULT_NUM_FACES = 20


def _collect_image_paths(images, images_dir):
    """
    Build a list of image paths from CLI arguments, keeping order and removing duplicates.
    """
    paths = []
    if images:
        paths.extend(images)
    if images_dir:
        supported_ext = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for ext in supported_ext:
            paths.extend(glob.glob(os.path.join(images_dir, ext)))
    # De‑duplicate while preserving order
    seen = set()
    ordered_unique = []
    for p in paths:
        if p not in seen:
            ordered_unique.append(p)
            seen.add(p)
    return ordered_unique


def main():
    parser = argparse.ArgumentParser(description='Facial Recognition System')
    parser.add_argument('--mode', choices=['stream', 'register', 'test'], default='stream',
                        help='Operation mode')
    parser.add_argument('--name', type=str, help='Person name for registration')
    parser.add_argument('--employee-id', type=str, help='Employee ID for registration')
    parser.add_argument('--department', type=str, help='Department for registration')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--video', type=str, help='Video file path (instead of camera)')
    parser.add_argument('--images', nargs='+', type=str,
                        help='One or more image files to use for registration')
    parser.add_argument('--images-dir', type=str,
                        help='Directory containing face images for registration')
    parser.add_argument('--num-faces', type=int, default=DEFAULT_NUM_FACES,
                        help=f'Number of face images to register (default: {DEFAULT_NUM_FACES})')
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

            face_images = []

            # -------- New flow: load faces from images --------
            if args.images or args.images_dir:
                image_paths = _collect_image_paths(args.images, args.images_dir)
                if len(image_paths) < args.num_faces:
                    print(f"Error: At least {args.num_faces} images are required, "
                          f"but only {len(image_paths)} were provided.")
                    sys.exit(1)

                print(f"Found {len(image_paths)} candidate images. "
                      f"Processing up to {args.num_faces} that contain exactly one face...")
                face_images = system.load_faces_from_files(image_paths, num_faces=args.num_faces)

                if len(face_images) < args.num_faces:
                    print(f"Error: Only {len(face_images)} valid face images were extracted "
                          f"(need {args.num_faces}). Registration cancelled.")
                    sys.exit(1)
            # --------------------------------------------------

            else:
                # Fallback: capture from live camera / video
                source = args.video if args.video else args.camera
                face_images = system.capture_faces_for_registration(source, num_faces=args.num_faces)
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

2. Register a new person **from images in a directory**:
   python example_usage.py --mode register --name "John Doe" --employee-id "EMP001" \
     --department "Engineering" --images-dir ./john_doe_faces

3. Register a new person **by specifying individual images**:
   python example_usage.py --mode register --name "John Doe" --employee-id "EMP001" \
     --images img1.jpg img2.jpg ... img20.jpg

4. Capture faces with the camera (fallback):
   python example_usage.py --mode register --name "John Doe" --employee-id "EMP001"

5. Use a video file instead of camera:
   python example_usage.py --mode stream --video path/to/video.mp4

6. Test system and show statistics:
   python example_usage.py --mode test

7. Use a different camera:
   python example_usage.py --mode stream --camera 1
"""
