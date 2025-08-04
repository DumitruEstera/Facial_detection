import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import queue
import time
import numpy as np
from datetime import datetime, timedelta
from facial_recognition_system import FacialRecognitionSystem
from license_plate_recognition_system import LicensePlateRecognitionSystem

# Import DeepFace for demographics analysis
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

class FaceDemographicsAnalyzer:
    """Class to handle face demographics analysis using DeepFace"""
    
    def __init__(self):
        self.enabled = DEEPFACE_AVAILABLE
        self.last_analysis_cache = {}  # Cache to avoid duplicate processing
        self.cache_duration = 3.0  # Analyze same face only once per 3 seconds
        self.debug_mode = False  # Enable debug output

    def analyze_face(self, face_image: np.ndarray, face_bbox: tuple) -> dict:
        """
        Analyze face for emotion, age, and gender
        
        Args:
            face_image: Cropped face image
            face_bbox: Bounding box (x, y, w, h) for caching
            
        Returns:
            Dict with emotion, age, gender info or empty dict
        """ 
        try:
            # Create cache key based on face position and current time
            x, y, w, h = face_bbox
            cache_key = f"{x}_{y}_{w}_{h}"
            current_time = time.time()
            
            # Check if we analyzed this face recently
            if cache_key in self.last_analysis_cache:
                last_time, last_result = self.last_analysis_cache[cache_key]
                if current_time - last_time < self.cache_duration:
                    return last_result
            
            # Ensure image is in the right format for DeepFace
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                # Convert BGR to RGB for DeepFace
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_image
                
            # Resize if too small (DeepFace works better with larger images)
            height, width = face_rgb.shape[:2]
            if width < 48 or height < 48:
                scale = max(48/width, 48/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                face_rgb = cv2.resize(face_rgb, (new_width, new_height))

            # Run DeepFace analysis
            analysis = DeepFace.analyze(
                img_path=face_rgb,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False,  # Don't crash if no face detected
                silent=True  # Suppress verbose output
            )
            

            # DeepFace returns a list, get first result
            if isinstance(analysis, list) and len(analysis) > 0:
                result = analysis[0]
            else:
                result = analysis
                
            # Extract relevant information
            demographics = {
                'age': int(result.get('age', 0)),
                'gender': result.get('dominant_gender', 'unknown'),
                'emotion': result.get('dominant_emotion', 'unknown'),
                'gender_confidence': result.get('gender', {}).get(result.get('dominant_gender', ''), 0),
                'emotion_confidence': result.get('emotion', {}).get(result.get('dominant_emotion', ''), 0)
            }

            # Cache the result
            self.last_analysis_cache[cache_key] = (current_time, demographics)
            
            # Clean old cache entries (keep cache size manageable)
            self._clean_cache(current_time)
            
            return demographics
            
        except Exception as e:
            print(f"❌ Error in face demographics analysis: {e}")
            import traceback
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            return {}
    
    def _clean_cache(self, current_time):
        """Remove old cache entries"""
        keys_to_remove = []
        for key, (timestamp, _) in self.last_analysis_cache.items():
            if current_time - timestamp > self.cache_duration * 2:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.last_analysis_cache[key]

class IntegratedSecurityGUI:
    def __init__(self, root, db_config):
        self.root = root
        self.root.title("Enhanced Security System - Face Recognition & Demographics")
        self.root.geometry("1600x900")
        
        # Initialize systems
        self.face_system = FacialRecognitionSystem(db_config, camera_id="0")
        self.plate_system = LicensePlateRecognitionSystem(db_config)
        self.demographics_analyzer = FaceDemographicsAnalyzer()
        
        # Video capture
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Current mode
        self.mode = tk.StringVar(value="both")  # "face", "plate", or "both"
        
        # Demographics analysis settings
        self.analyze_demographics = tk.BooleanVar(value=True)
        self.demographics_test_mode = tk.BooleanVar(value=False)  # For testing on known faces too
        
        # Setup GUI
        self.setup_gui()
        
        # Start video thread
        self.video_thread = None
        
    def setup_gui(self):
        """Setup the enhanced GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Top control panel
        control_panel = ttk.Frame(main_frame)
        control_panel.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Mode selection
        ttk.Label(control_panel, text="Detection Mode:").grid(row=0, column=0, padx=5)
        
        ttk.Radiobutton(control_panel, text="Face Recognition", 
                       variable=self.mode, value="face").grid(row=0, column=1, padx=5)
        ttk.Radiobutton(control_panel, text="License Plate", 
                       variable=self.mode, value="plate").grid(row=0, column=2, padx=5)
        ttk.Radiobutton(control_panel, text="Both", 
                       variable=self.mode, value="both").grid(row=0, column=3, padx=5)
        
        # Demographics analysis toggle
        ttk.Checkbutton(control_panel, text="🧠 Analyze Demographics (Age/Gender/Emotion)", 
                       variable=self.analyze_demographics).grid(row=0, column=4, padx=20)
        
        # Test mode toggle (analyze all faces, not just unknown ones)
        ttk.Checkbutton(control_panel, text="🔬 Test Mode (Analyze All Faces)", 
                       variable=self.demographics_test_mode).grid(row=0, column=5, padx=10)
        
        # Camera controls
        self.start_btn = ttk.Button(control_panel, text="Start Camera", 
                                   command=self.start_camera)
        self.start_btn.grid(row=0, column=6, padx=20)
        
        self.stop_btn = ttk.Button(control_panel, text="Stop Camera", 
                                  command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=7, padx=5)
        
        # Left panel - Video feed
        video_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding="10")
        video_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.grid(row=0, column=0)
        
        # Status label for demographics
        self.demographics_status = ttk.Label(video_frame, text="")
        self.demographics_status.grid(row=1, column=0, pady=5)
        
        # Right panel - Tabbed interface
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Face Recognition Tab
        self.face_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.face_tab, text="Face Recognition")
        self.setup_enhanced_face_tab()
        
        # License Plate Tab
        self.plate_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.plate_tab, text="License Plates")
        self.setup_plate_tab()
        
        # Statistics Tab
        self.stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_tab, text="Statistics")
        self.setup_stats_tab()
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(1, weight=1)
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        
        # Update demographics status
        self.update_demographics_status()
        
    def update_demographics_status(self):
        """Update the demographics analysis status display"""
        if self.demographics_analyzer.enabled:
            if self.analyze_demographics.get():
                test_mode_text = " + TEST MODE" if self.demographics_test_mode.get() else ""
                self.demographics_status.config(text=f"🧠 Demographics Analysis: ENABLED{test_mode_text}", 
                                              foreground="green")
            else:
                self.demographics_status.config(text="🧠 Demographics Analysis: DISABLED", 
                                              foreground="orange")
        else:
            self.demographics_status.config(text="❌ DeepFace not available - Install: pip install deepface", 
                                          foreground="red")
        
        # Schedule next update
        self.root.after(1000, self.update_demographics_status)
        
    def setup_enhanced_face_tab(self):
        """Setup enhanced face recognition tab with demographics"""
        # Face recognition log
        log_label = ttk.Label(self.face_tab, text="Face Recognition & Demographics Log:")
        log_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        
        # Create treeview for log with additional columns
        columns = ('Time', 'Name', 'ID', 'Confidence', 'Age', 'Gender', 'Emotion')
        self.face_log_tree = ttk.Treeview(self.face_tab, 
                                         columns=columns, 
                                         show='tree headings', height=12)
        self.face_log_tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)
        
        # Configure columns
        self.face_log_tree.column('#0', width=0, stretch=tk.NO)
        self.face_log_tree.column('Time', width=120)
        self.face_log_tree.column('Name', width=120)
        self.face_log_tree.column('ID', width=80)
        self.face_log_tree.column('Confidence', width=80)
        self.face_log_tree.column('Age', width=50)
        self.face_log_tree.column('Gender', width=70)
        self.face_log_tree.column('Emotion', width=90)
        
        self.face_log_tree.heading('Time', text='Time')
        self.face_log_tree.heading('Name', text='Name')
        self.face_log_tree.heading('ID', text='ID')
        self.face_log_tree.heading('Confidence', text='Conf.')
        self.face_log_tree.heading('Age', text='Age')
        self.face_log_tree.heading('Gender', text='Gender')
        self.face_log_tree.heading('Emotion', text='Emotion')
        
        # Scrollbar
        face_scrollbar = ttk.Scrollbar(self.face_tab, orient=tk.VERTICAL, 
                                      command=self.face_log_tree.yview)
        face_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.face_log_tree.configure(yscrollcommand=face_scrollbar.set)
        
        # Face registration section (simplified)
        reg_frame = ttk.LabelFrame(self.face_tab, text="Face Registration", padding="10")
        reg_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        
        ttk.Label(reg_frame, text="Name:").grid(row=0, column=0, sticky=tk.W)
        self.face_name_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.face_name_var, width=25).grid(row=0, column=1, padx=5)
        
        ttk.Label(reg_frame, text="Employee ID:").grid(row=1, column=0, sticky=tk.W)
        self.face_emp_id_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.face_emp_id_var, width=25).grid(row=1, column=1, padx=5)
        
        ttk.Button(reg_frame, text="Register Face", 
                  command=self.register_face).grid(row=2, column=0, columnspan=2, pady=10)
        
    def setup_plate_tab(self):
        """Setup license plate tab (unchanged from original)"""
        # License plate log
        log_label = ttk.Label(self.plate_tab, text="License Plate Log:")
        log_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        
        # Create treeview for log
        self.plate_log_tree = ttk.Treeview(self.plate_tab, 
                                          columns=('Time', 'Plate', 'Vehicle', 'Status', 'Confidence'), 
                                          show='tree headings', height=10)
        self.plate_log_tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)
        
        # Configure columns
        self.plate_log_tree.column('#0', width=0, stretch=tk.NO)
        self.plate_log_tree.column('Time', width=150)
        self.plate_log_tree.column('Plate', width=100)
        self.plate_log_tree.column('Vehicle', width=80)
        self.plate_log_tree.column('Status', width=100)
        self.plate_log_tree.column('Confidence', width=80)
        
        self.plate_log_tree.heading('Time', text='Time')
        self.plate_log_tree.heading('Plate', text='Plate Number')
        self.plate_log_tree.heading('Vehicle', text='Vehicle')
        self.plate_log_tree.heading('Status', text='Status')
        self.plate_log_tree.heading('Confidence', text='Conf.')
        
        # Scrollbar
        plate_scrollbar = ttk.Scrollbar(self.plate_tab, orient=tk.VERTICAL, 
                                       command=self.plate_log_tree.yview)
        plate_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.plate_log_tree.configure(yscrollcommand=plate_scrollbar.set)
        
        # Plate registration section
        reg_frame = ttk.LabelFrame(self.plate_tab, text="License Plate Registration", padding="10")
        reg_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        
        ttk.Label(reg_frame, text="Plate Number:").grid(row=0, column=0, sticky=tk.W)
        self.plate_number_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.plate_number_var, width=25).grid(row=0, column=1, padx=5)
        
        ttk.Label(reg_frame, text="Owner Name:").grid(row=1, column=0, sticky=tk.W)
        self.plate_owner_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.plate_owner_var, width=25).grid(row=1, column=1, padx=5)
        
        ttk.Button(reg_frame, text="Register Plate", 
                  command=self.register_plate).grid(row=2, column=0, columnspan=2, pady=10)
        
    def setup_stats_tab(self):
        """Setup statistics tab (unchanged from original)"""
        # Create frame for statistics
        stats_frame = ttk.Frame(self.stats_tab, padding="20")
        stats_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(stats_frame, text="System Statistics", 
                               font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Statistics labels
        self.stats_labels = {}
        
        stats_items = [
            ('total_persons', 'Total Registered Persons:'),
            ('total_face_embeddings', 'Total Face Embeddings:'),
            ('total_face_accesses', 'Total Face Accesses:'),
            ('', ''),  # Separator
            ('total_plates', 'Total License Plates:'),
            ('authorized_plates', 'Authorized Plates:'),
            ('total_vehicle_accesses', 'Total Vehicle Accesses:'),
            ('unauthorized_vehicle_accesses', 'Unauthorized Accesses:')
        ]
        
        row = 1
        for key, label_text in stats_items:
            if key == '':
                row += 1
                continue
                
            ttk.Label(stats_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2)
            self.stats_labels[key] = ttk.Label(stats_frame, text="0")
            self.stats_labels[key].grid(row=row, column=1, sticky=tk.W, padx=20, pady=2)
            row += 1
            
        # Refresh button
        ttk.Button(stats_frame, text="Refresh Statistics", 
                  command=self.update_statistics).grid(row=row+1, column=0, columnspan=2, pady=20)
        
        # Initial update
        self.update_statistics()
        
    def start_camera(self):
        """Start the camera feed"""

        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.start_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.NORMAL)

                # Start video processing thread
                self.video_thread = threading.Thread(target=self.process_video)
                self.video_thread.daemon = True
                self.video_thread.start()
                
                # Start display update
                self.update_display()
            else:
                messagebox.showerror("Error", "Could not open camera")


    def stop_camera(self):
        """Stop the camera feed"""

        self.is_running = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def process_video(self):
        """Enhanced video processing with demographics analysis"""

        frame_count = 0
        
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame_count += 1
                    mode = self.mode.get()
                    face_results = []
                    plate_results = []
                    annotated_frame = frame.copy()
                

                    # Process based on mode
                    if mode in ['face', 'both']:
                        # Get the basic face recognition results
                        face_frame, face_results = self.face_system.process_frame(frame)
                        
                        # ENHANCED: Also detect faces that weren't recognized and add them as "Unknown"
                        all_faces = self.face_system.face_detector.detect_faces(frame)
                        
                        # For any face that wasn't recognized, add it as "Unknown"
                        recognized_bboxes = [r.get('bbox') for r in face_results if 'bbox' in r]
                        
                        for face_bbox in all_faces:
                            # Check if this face was already recognized
                            face_already_processed = False
                            for rec_bbox in recognized_bboxes:
                                if rec_bbox and self._bbox_overlap(face_bbox, rec_bbox) > 0.5:
                                    face_already_processed = True
                                    break
                            
                            # If face wasn't recognized, add it as "Unknown"
                            if not face_already_processed:
                                unknown_result = {
                                    'name': 'Unknown',
                                    'employee_id': 'Unknown',
                                    'department': None,
                                    'confidence': 0.0,
                                    'bbox': face_bbox,
                                    'timestamp': datetime.now(),
                                    'person_id': None
                                }
                                face_results.append(unknown_result)
          
                        # Enhanced: Add demographics analysis
                        if self.analyze_demographics.get() and self.demographics_analyzer.enabled:
                            face_results = self.enhance_face_results_with_demographics(frame, face_results)
                        
                        if mode == 'face':
                            # Redraw the frame with all faces (including unknown ones)
                            annotated_frame = self._draw_all_faces(frame.copy(), face_results)
                            
                    if mode in ['plate', 'both']:
                        plate_results = self.plate_system.process_frame(frame)
                        plate_frame = self.plate_system.draw_outputs(frame.copy(), plate_results)
                        if mode == 'plate':
                            annotated_frame = plate_frame
                            
                    if mode == 'both':
                        # Draw both faces and plates
                        annotated_frame = self._draw_all_faces(plate_frame, face_results)
                        
                    # Add to queue
                    try:
                        self.frame_queue.put((annotated_frame, face_results, plate_results), 
                                           block=False)
                    except queue.Full:
                        pass

    def _bbox_overlap(self, bbox1, bbox2):
        """Calculate overlap ratio between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _draw_all_faces(self, frame, face_results):
        """Draw all faces including demographics info"""
        for result in face_results:
            if 'bbox' not in result:
                continue
                
            x, y, w, h = result['bbox']
            
            # Choose color based on recognition status
            if result.get('name') == 'Unknown':
                color = (0, 0, 255)  # Red for unknown
            else:
                color = (0, 255, 0)  # Green for known
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label with demographics if available
            label_lines = []
            label_lines.append(f"{result.get('name', 'Unknown')}")
            
            if result.get('confidence', 0) > 0:
                label_lines.append(f"Conf: {result['confidence']:.2f}")
            
            # Add demographics for unknown faces
            if result.get('demographics_analyzed'):
                if result.get('age'):
                    label_lines.append(f"Age: {result['age']}")
                if result.get('gender'):
                    label_lines.append(f"Gender: {result['gender']}")
                if result.get('emotion'):
                    label_lines.append(f"Emotion: {result['emotion']}")
            
            # Draw labels
            for i, line in enumerate(label_lines):
                y_offset = y - 5 - (len(label_lines) - 1 - i) * 20
                if y_offset < 20:
                    y_offset = y + h + 20 + i * 20
                    
                cv2.putText(frame, line, (x, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
                        
    def enhance_face_results_with_demographics(self, frame, face_results):
        """Add demographics analysis to face recognition results"""
        if not face_results:

            return []
            
        enhanced_results = []
        
        for i, result in enumerate(face_results):
            # Copy the original result
            enhanced_result = result.copy()
            
            # Determine if we should analyze this face
            is_unknown = (result.get('name') == 'Unknown' or 
                         result.get('confidence', 0) < self.face_system.min_confidence)
            
            # In test mode, analyze all faces; otherwise only unknown ones
            should_analyze = (self.demographics_test_mode.get() or is_unknown)
            
            if should_analyze and 'bbox' in result:
                # Extract face from frame
                x, y, w, h = result['bbox']

                face_crop = frame[y:y+h, x:x+w]
                
                if face_crop.size > 0:

                    # Analyze demographics
                    demographics = self.demographics_analyzer.analyze_face(face_crop, result['bbox'])
                    
                    if demographics:  # If analysis was successful
                        enhanced_result.update({
                            'age': demographics.get('age', 'N/A'),
                            'gender': demographics.get('gender', 'unknown'),
                            'emotion': demographics.get('emotion', 'unknown'),
                            'demographics_analyzed': True
                        })
                        
                    else:

                        # Analysis failed, add empty demographics
                        enhanced_result.update({
                            'age': 'N/A',
                            'gender': 'N/A',
                            'emotion': 'N/A',
                            'demographics_analyzed': False
                        })
                else:

                    # Empty face crop
                    enhanced_result.update({
                        'age': 'N/A',
                        'gender': 'N/A',
                        'emotion': 'N/A',
                        'demographics_analyzed': False
                    })
            else:
                # For known faces (when not in test mode), don't analyze demographics
                enhanced_result.update({
                    'age': '',
                    'gender': '',
                    'emotion': '',
                    'demographics_analyzed': False
                })
                
            enhanced_results.append(enhanced_result)
            
        return enhanced_results
        
    def update_display(self):
        """Update the video display"""
        try:
            frame, face_results, plate_results = self.frame_queue.get(block=False)
            
            # Convert frame to RGB and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (800, 600))
            
            # Convert to PIL Image
            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image=image)
            
            # Update label
            self.video_label.config(image=photo)
            self.video_label.image = photo
            
            # Update logs
            for result in face_results:
                self.add_to_enhanced_face_log(result)
                
            for result in plate_results:
                self.add_to_plate_log(result)
                
        except queue.Empty:
            pass
            
        if self.is_running:
            self.root.after(30, self.update_display)
            
    def add_to_enhanced_face_log(self, result):
        """Add enhanced face recognition result to log"""
        time_str = result['timestamp'].strftime("%H:%M:%S")
        confidence_str = f"{result['confidence']:.2f}" if 'confidence' in result else "N/A"
        
        # Get demographics info
        age = str(result.get('age', ''))
        gender = result.get('gender', '')
        emotion = result.get('emotion', '')
        
        # Format demographics for display
        if result.get('demographics_analyzed', False):
            age_display = age if age != 'N/A' else ''
            gender_display = gender.capitalize() if gender != 'unknown' else ''
            emotion_display = emotion.capitalize() if emotion != 'unknown' else ''
        else:
            age_display = ''
            gender_display = ''
            emotion_display = ''
        
        # Insert into tree
        values = (time_str, result['name'], result['employee_id'], 
                 confidence_str, age_display, gender_display, emotion_display)
        
        item = self.face_log_tree.insert('', 0, values=values)
        
        # Color code unknown faces with demographics
        if result.get('name') == 'Unknown' and result.get('demographics_analyzed', False):
            self.face_log_tree.set(item, 'Name', '👤 Unknown')
            
        # Keep only last 100 entries
        children = self.face_log_tree.get_children()
        if len(children) > 100:
            self.face_log_tree.delete(children[-1])
            
    def add_to_plate_log(self, result):
        """Add license plate result to log (unchanged)"""
        time_str = result['timestamp'].strftime("%H:%M:%S")
        status = "AUTHORIZED" if result['is_authorized'] else "UNAUTHORIZED"
        confidence_str = f"{result['confidence']:.2f}"
        
        self.plate_log_tree.insert('', 0, values=(time_str, result['plate_number'],
                                                 result['vehicle_type'], status, confidence_str))
        
        # Keep only last 100 entries
        children = self.plate_log_tree.get_children()
        if len(children) > 100:
            self.plate_log_tree.delete(children[-1])
            
    def register_face(self):
        """Handle face registration (unchanged)"""
        name = self.face_name_var.get().strip()
        emp_id = self.face_emp_id_var.get().strip()
        
        if not name or not emp_id:
            messagebox.showerror("Error", "Name and Employee ID are required")
            return
            
        messagebox.showinfo("Info", "Please use the example_usage.py script for face registration")
            
    def register_plate(self):
        """Handle license plate registration (simplified)"""
        plate_number = self.plate_number_var.get().strip()
        owner_name = self.plate_owner_var.get().strip()
        
        if not plate_number:
            messagebox.showerror("Error", "Plate number is required")
            return
            
        try:
            success = self.plate_system.register_plate(
                plate_number=plate_number,
                owner_name=owner_name,
                is_authorized=True
            )
            
            if success:
                messagebox.showinfo("Success", f"Successfully registered plate: {plate_number}")
                self.plate_number_var.set("")
                self.plate_owner_var.set("")
            else:
                messagebox.showerror("Error", "Failed to register plate")
                
        except Exception as e:
            messagebox.showerror("Error", f"Registration failed: {str(e)}")
            
    def update_statistics(self):
        """Update system statistics display (unchanged)"""
        stats = self.face_system.db.get_statistics()
        
        for key, label in self.stats_labels.items():
            if key in stats:
                label.config(text=str(stats[key]))
                
    def on_closing(self):
        """Handle window closing"""
        self.stop_camera()
        self.face_system.cleanup()
        self.root.destroy()


def main():
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'facial_recognition_db',
        'user': 'postgres',
        'password': 'admin'  # Change this
    }

    try:
        # Create GUI
        root = tk.Tk()
        app = IntegratedSecurityGUI(root, db_config)
        
        # Handle window closing
        root.protocol("WM_DELETE_WINDOW", app.on_closing)

        # Start GUI
        root.mainloop()
        
    except Exception as e:

        import traceback


if __name__ == "__main__":
    main()