# Facial Recognition Security System

A comprehensive security system based on real-time video stream analysis and facial recognition for identifying authorized personnel and detecting potential security threats.

## 🎯 Project Overview

This application enhances security through real-time analysis of video recordings from surveillance camera networks to detect and evaluate potential security risks. The system can identify unauthorized persons, monitor restricted access zones, and maintain detailed access logs.

## 🚀 Features Implemented

### Core Functionality
- **Real-time Face Detection**: Uses YuNet face detector for accurate face detection in video streams
- **Face Recognition**: Employs FaceNet-based feature extraction with FAISS similarity search for fast person identification
- **Person Registration**: Complete workflow for registering new personnel with multiple face samples
- **Access Control**: Monitors and logs all access events with confidence scores
- **Database Integration**: Persistent storage of person data, face embeddings, and access logs

### User Interfaces
- **GUI Application**: Full-featured Tkinter interface with live video feed, recognition logs, and registration capabilities
- **CLI Interface**: Command-line tool for various operations (streaming, registration, testing)
- **Real-time Monitoring**: Live video display with face bounding boxes and identification labels

### Advanced Features
- **Threaded Processing**: Concurrent video processing for smooth real-time performance
- **Confidence Scoring**: Adjustable thresholds for recognition accuracy
- **Batch Processing**: Efficient handling of multiple face embeddings
- **Statistics Dashboard**: System metrics and performance monitoring

## 🛠️ Technologies Used

### Computer Vision & AI
- **OpenCV (cv2)**: Video capture, image processing, and computer vision operations
- **YuNet**: State-of-the-art face detection model (OpenCV's face detector)
- **FaceNet**: Deep learning model for face feature extraction (128-dimensional embeddings)
- **TensorFlow/Keras**: Deep learning framework for neural network operations

### Database & Search
- **PostgreSQL**: Relational database for storing person information and access logs
- **psycopg2**: PostgreSQL database adapter for Python
- **FAISS (Facebook AI Similarity Search)**: High-performance vector similarity search for fast face matching

### User Interface
- **Tkinter**: Native Python GUI framework for desktop application
- **PIL (Pillow)**: Image processing library for GUI image display
- **Threading & Queue**: Concurrent programming for responsive UI

### Data Processing
- **NumPy**: Numerical computing for array operations and embeddings
- **Pickle**: Serialization for storing embeddings in database

## 📁 Project Structure

```
Facial_detection/
├── facial_recognition_system.py    # Main system orchestrator
├── face_detection.py              # YuNet face detection implementation
├── feature_extraction.py          # FaceNet feature extraction
├── faiss_index.py                 # FAISS vector search implementation
├── database_manager.py            # PostgreSQL database operations
├── gui_application.py             # Tkinter GUI application
├── example_usage.py               # CLI interface and usage examples
└── README.md                      # Project documentation
```

## 🏗️ Architecture

### Core Components

1. **FacialRecognitionSystem**: Main orchestrator class that coordinates all components
2. **YuNetFaceDetector**: Handles face detection in video frames
3. **FaceNetFeatureExtractor**: Extracts 128-dimensional face embeddings
4. **FaissIndex**: Manages vector similarity search for face matching
5. **DatabaseManager**: Handles all database operations and data persistence
6. **FacialRecognitionGUI**: Provides user-friendly interface for system interaction

### Data Flow

1. **Video Capture** → Face Detection → Feature Extraction → Database Search → Recognition Result
2. **Registration Flow** → Face Capture → Feature Extraction → Database Storage → Index Update

## ⚙️ Key Implementation Details

### Face Detection
- Uses YuNet model for robust face detection
- Configurable confidence and NMS thresholds
- Automatic model downloading and setup

### Feature Extraction
- FaceNet-based 128-dimensional embeddings
- L2 normalization for consistent similarity metrics
- Batch processing support for efficiency

### Vector Search
- FAISS FlatL2 index for exact similarity search
- Configurable distance thresholds for recognition
- Person ID mapping for database integration

### Database Schema
- **persons**: Store person information (name, employee_id, department, authorized_zones)
- **face_embeddings**: Store face feature vectors linked to persons
- **access_logs**: Track all recognition events with timestamps and confidence scores

### Performance Optimizations
- Threaded video processing to maintain smooth frame rates
- Queue-based frame buffering
- Batch embedding extraction
- Efficient FAISS indexing for fast search

## 🎮 Usage Examples

### CLI Interface
```bash
# Start live facial recognition
python example_usage.py --mode stream

# Register a new person
python example_usage.py --mode register --name "John Doe" --employee-id "EMP001"

# Process video file
python example_usage.py --mode stream --video path/to/video.mp4

# Show system statistics
python example_usage.py --mode test
```

### GUI Application
```bash
python gui_application.py
```

## 🔧 Setup Requirements

### Dependencies
- Python 3.7+
- OpenCV with contrib modules
- TensorFlow 2.x
- PostgreSQL database
- FAISS library
- Required Python packages (see imports in source files)

### Database Setup
- PostgreSQL server running
- Database schema creation required
- Configure connection parameters in code

## 🎯 Current Status

**Implemented:**
- ✅ Complete face detection pipeline
- ✅ Face recognition with confidence scoring
- ✅ Database integration and persistence
- ✅ Real-time video processing
- ✅ Person registration workflow
- ✅ GUI and CLI interfaces
- ✅ Access logging and monitoring

**Note:** The current FaceNet implementation uses a placeholder model. For production use, replace with a pre-trained FaceNet model for optimal recognition accuracy.

