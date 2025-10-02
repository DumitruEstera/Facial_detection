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
from fire_detection_system import FireDetectionSystem

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
        self.last_analysis_cache = {}
        self.cache_duration = 3.0
        self.debug_mode = False

    def analyze_face(self, face_image: np.ndarray, face_bbox: tuple) -> dict:
        """Analyze face for emotion, age, and gender"""
        try:
            x, y, w, h = face_bbox
            cache_key = f"{x}_{y}_{w}_{h}"
            current_time = time.time()
            
            if cache_key in self.last_analysis_cache:
                last_time, last_result = self.last_analysis_cache[cache_key]
                if current_time - last_time < self.cache_duration:
                    return last_result
            
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_image
                
            height, width = face_rgb.shape[:2]
            if width < 48 or height < 48:
                scale = max(48/width, 48/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                face_rgb = cv2.resize(face_rgb, (new_width, new_height))

            analysis = DeepFace.analyze(
                img_path=face_rgb,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(analysis, list) and len(analysis) > 0:
                result = analysis[0]
            else:
                result = analysis
                
            demographics = {
                'age': int(result.get('age', 0)),
                'gender': result.get('dominant_gender', 'unknown'),
                'emotion': result.get('dominant_emotion', 'unknown'),
                'gender_confidence': result.get('gender', {}).get(result.get('dominant_gender', ''), 0),
                'emotion_confidence': result.get('emotion', {}).get(result.get('dominant_emotion', ''), 0)
            }

            self.last_analysis_cache[cache_key] = (current_time, demographics)
            self._clean_cache(current_time)
            
            return demographics
            
        except Exception as e:
            print(f"❌ Error in face demographics analysis: {e}")
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
        self.root.title("Enhanced Security System - Face, Plate & Fire Detection")
        self.root.geometry("1600x900")
        
        # Initialize systems
        self.face_system = FacialRecognitionSystem(db_config, camera_id="0")
        self.plate_system = LicensePlateRecognitionSystem(db_config)
        self.demographics_analyzer = FaceDemographicsAnalyzer()
        
        # Initialize fire detection system
        try:
            self.fire_system = FireDetectionSystem()
            self.fire_enabled = True
            print("✅ Fire detection system initialized")
        except Exception as e:
            print(f"⚠️ Fire detection not available: {e}")
            self.fire_enabled = False
            messagebox.showwarning(
                "Fire Detection Unavailable",
                f"Fire detection could not be initialized:\n{e}\n\n"
                "Please download the model from:\n"
                "https://huggingface.co/kittendev/YOLOv8m-smoke-detection\n"
                "and place it as 'models/fire_detection.pt'"
            )
        
        # Video capture
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Current mode - now includes fire detection
        self.mode = tk.StringVar(value="all")  # "face", "plate", "fire", "all"
        
        # Demographics analysis settings
        self.analyze_demographics = tk.BooleanVar(value=True)
        self.demographics_test_mode = tk.BooleanVar(value=False)
        
        # Setup GUI
        self.setup_gui()
        
        # Start video thread
        self.video_thread = None
        
    def setup_gui(self):
        """Setup the enhanced GUI layout with fire detection"""
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
        ttk.Radiobutton(control_panel, text="🔥 Fire Detection", 
                       variable=self.mode, value="fire", 
                       state=tk.NORMAL if self.fire_enabled else tk.DISABLED).grid(row=0, column=3, padx=5)
        ttk.Radiobutton(control_panel, text="All Systems", 
                       variable=self.mode, value="all").grid(row=0, column=4, padx=5)
        
        # Demographics analysis toggle
        ttk.Checkbutton(control_panel, text="🧠 Analyze Demographics", 
                       variable=self.analyze_demographics).grid(row=0, column=5, padx=20)
        
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
        
        # Status label
        self.status_label = ttk.Label(video_frame, text="", font=('Arial', 10, 'bold'))
        self.status_label.grid(row=1, column=0, pady=5)
        
        # Right panel - Tabbed interface
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Face Recognition Tab
        self.face_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.face_tab, text="Face Recognition")
        self.setup_face_tab()
        
        # License Plate Tab
        self.plate_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.plate_tab, text="License Plates")
        self.setup_plate_tab()
        
        # Fire Detection Tab
        self.fire_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.fire_tab, text="🔥 Fire Detection")
        self.setup_fire_tab()
        
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
        
    def setup_face_tab(self):
        """Setup face recognition tab"""
        log_label = ttk.Label(self.face_tab, text="Face Recognition Log:")
        log_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        
        columns = ('Time', 'Name', 'ID', 'Confidence', 'Age', 'Gender', 'Emotion')
        self.face_log_tree = ttk.Treeview(self.face_tab, 
                                         columns=columns, 
                                         show='tree headings', height=12)
        self.face_log_tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)
        
        self.face_log_tree.column('#0', width=0, stretch=tk.NO)
        self.face_log_tree.column('Time', width=120)
        self.face_log_tree.column('Name', width=120)
        self.face_log_tree.column('ID', width=80)
        self.face_log_tree.column('Confidence', width=80)
        self.face_log_tree.column('Age', width=50)
        self.face_log_tree.column('Gender', width=70)
        self.face_log_tree.column('Emotion', width=90)
        
        for col in columns:
            self.face_log_tree.heading(col, text=col)
        
        scrollbar = ttk.Scrollbar(self.face_tab, orient=tk.VERTICAL, 
                                 command=self.face_log_tree.yview)
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.face_log_tree.configure(yscrollcommand=scrollbar.set)
        
    def setup_plate_tab(self):
        """Setup license plate tab"""
        log_label = ttk.Label(self.plate_tab, text="License Plate Log:")
        log_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        
        self.plate_log_tree = ttk.Treeview(self.plate_tab, 
                                          columns=('Time', 'Plate', 'Vehicle', 'Status', 'Confidence'), 
                                          show='tree headings', height=15)
        self.plate_log_tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)
        
        self.plate_log_tree.column('#0', width=0, stretch=tk.NO)
        self.plate_log_tree.column('Time', width=150)
        self.plate_log_tree.column('Plate', width=100)
        self.plate_log_tree.column('Vehicle', width=80)
        self.plate_log_tree.column('Status', width=100)
        self.plate_log_tree.column('Confidence', width=80)
        
        for col in ('Time', 'Plate', 'Vehicle', 'Status', 'Confidence'):
            self.plate_log_tree.heading(col, text=col)
        
        scrollbar = ttk.Scrollbar(self.plate_tab, orient=tk.VERTICAL, 
                                 command=self.plate_log_tree.yview)
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.plate_log_tree.configure(yscrollcommand=scrollbar.set)
        
    def setup_fire_tab(self):
        """Setup fire detection tab"""
        # Title
        title_label = ttk.Label(self.fire_tab, text="🔥 Fire & Smoke Detection Log", 
                               font=('Arial', 12, 'bold'))
        title_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        
        # Fire detection log
        columns = ('Time', 'Type', 'Confidence', 'Severity', 'Alert')
        self.fire_log_tree = ttk.Treeview(self.fire_tab, 
                                         columns=columns, 
                                         show='tree headings', height=12)
        self.fire_log_tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)
        
        self.fire_log_tree.column('#0', width=0, stretch=tk.NO)
        self.fire_log_tree.column('Time', width=150)
        self.fire_log_tree.column('Type', width=100)
        self.fire_log_tree.column('Confidence', width=100)
        self.fire_log_tree.column('Severity', width=100)
        self.fire_log_tree.column('Alert', width=100)
        
        self.fire_log_tree.heading('Time', text='Time')
        self.fire_log_tree.heading('Type', text='Type')
        self.fire_log_tree.heading('Confidence', text='Confidence')
        self.fire_log_tree.heading('Severity', text='Severity')
        self.fire_log_tree.heading('Alert', text='Alert Status')
        
        scrollbar = ttk.Scrollbar(self.fire_tab, orient=tk.VERTICAL, 
                                 command=self.fire_log_tree.yview)
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.fire_log_tree.configure(yscrollcommand=scrollbar.set)
        
        # Alert statistics
        stats_frame = ttk.LabelFrame(self.fire_tab, text="Alert Statistics", padding="10")
        stats_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        
        self.fire_stats_labels = {}
        stats_items = [
            ('total_detections', 'Total Detections:'),
            ('fire_detections', 'Fire Detections:'),
            ('smoke_detections', 'Smoke Detections:'),
            ('critical_alerts', 'Critical Alerts:')
        ]
        
        for i, (key, label_text) in enumerate(stats_items):
            ttk.Label(stats_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.fire_stats_labels[key] = ttk.Label(stats_frame, text="0", 
                                                    font=('Arial', 10, 'bold'))
            self.fire_stats_labels[key].grid(row=i, column=1, sticky=tk.W, padx=20, pady=2)
        
        # Information box
        info_frame = ttk.LabelFrame(self.fire_tab, text="ℹ️ Information", padding="10")
        info_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=10, pady=5)
        
        info_text = ("Fire detection uses YOLOv8 model trained for smoke and fire detection.\n"
                    "Severity levels: LOW, MEDIUM, HIGH, CRITICAL\n"
                    "Alerts are triggered based on confidence and size of detection.")
        ttk.Label(info_frame, text=info_text, wraplength=400).grid(row=0, column=0, sticky=tk.W)
        
    def setup_stats_tab(self):
        """Setup statistics tab"""
        stats_frame = ttk.Frame(self.stats_tab, padding="20")
        stats_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        title_label = ttk.Label(stats_frame, text="System Statistics", 
                               font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        self.stats_labels = {}
        
        stats_items = [
            ('total_persons', 'Total Registered Persons:'),
            ('total_face_embeddings', 'Total Face Embeddings:'),
            ('total_face_accesses', 'Total Face Accesses:'),
            ('', ''),
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
            
        ttk.Button(stats_frame, text="Refresh Statistics", 
                  command=self.update_statistics).grid(row=row+1, column=0, columnspan=2, pady=20)
        
        self.update_statistics()
        
    def start_camera(self):
        """Start the camera feed"""
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.start_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.NORMAL)
                
                self.video_thread = threading.Thread(target=self.process_video)
                self.video_thread.daemon = True
                self.video_thread.start()
                
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
        """Video processing with fire detection"""
        frame_count = 0
        
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame_count += 1
                    mode = self.mode.get()
                    face_results = []
                    plate_results = []
                    fire_results = []
                    annotated_frame = frame.copy()
                    
                    try:
                        # Process based on mode
                        if mode in ['face', 'all']:
                            face_frame, face_results = self.face_system.process_frame(frame)
                            if mode == 'face':
                                annotated_frame = face_frame
                                
                        if mode in ['plate', 'all']:
                            plate_results = self.plate_system.process_frame(frame)
                            if plate_results:
                                if mode == 'plate':
                                    annotated_frame = self.plate_system.draw_outputs(frame.copy(), plate_results)
                                else:
                                    annotated_frame = self.plate_system.draw_outputs(annotated_frame, plate_results)
                        
                        # Fire detection
                        if mode in ['fire', 'all'] and self.fire_enabled:
                            fire_results = self.fire_system.process_frame(frame)
                            if fire_results:
                                if mode == 'fire':
                                    annotated_frame = self.fire_system.draw_detections(frame.copy(), fire_results)
                                else:
                                    annotated_frame = self.fire_system.draw_detections(annotated_frame, fire_results)
                        
                        # Add to queue
                        try:
                            self.frame_queue.put((annotated_frame, face_results, plate_results, fire_results), 
                                               block=False)
                        except queue.Full:
                            pass
                            
                    except Exception as e:
                        print(f"❌ Error processing frame: {e}")
                        try:
                            self.frame_queue.put((frame, [], [], []), block=False)
                        except queue.Full:
                            pass
                    
                    time.sleep(0.033)  # ~30 FPS
                    

    def update_display(self):
        """Update the video display"""
        try:
            frame, face_results, plate_results, fire_results = self.frame_queue.get(block=False)
            
            # Convert frame to RGB and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (800, 600))
            
            # Convert to PIL Image
            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image=image)
            
            # Update label
            self.video_label.config(image=photo)
            self.video_label.image = photo
            
            # Update status label
            status_text = ""
            if fire_results:
                critical_fires = [f for f in fire_results if f['severity'] == 'critical']
                if critical_fires:
                    status_text = "🚨 CRITICAL FIRE ALERT!"
                    self.status_label.config(foreground='red')
                else:
                    high_severity = [f for f in fire_results if f['severity'] in ['high', 'critical']]
                    if high_severity:
                        status_text = "⚠️ Fire/Smoke Detected"
                        self.status_label.config(foreground='orange')
            
            self.status_label.config(text=status_text)
            
            # Update logs
            for result in face_results:
                self.add_to_face_log(result)
                
            for result in plate_results:
                self.add_to_plate_log(result)
            
            for result in fire_results:
                self.add_to_fire_log(result)
                
        except queue.Empty:
            pass
            
        if self.is_running:
            self.root.after(30, self.update_display)
    
    def add_to_face_log(self, result):
        """Add face recognition result to log"""
        time_str = result['timestamp'].strftime("%H:%M:%S")
        confidence_str = f"{result['confidence']:.2f}" if 'confidence' in result else "N/A"
        
        age = str(result.get('age', ''))
        gender = result.get('gender', '')
        emotion = result.get('emotion', '')
        
        values = (time_str, result['name'], result['employee_id'], 
                 confidence_str, age, gender, emotion)
        
        self.face_log_tree.insert('', 0, values=values)
        
        children = self.face_log_tree.get_children()
        if len(children) > 100:
            self.face_log_tree.delete(children[-1])
            
    def add_to_plate_log(self, result):
        """Add license plate result to log"""
        time_str = result['timestamp'].strftime("%H:%M:%S")
        status = "AUTHORIZED" if result['is_authorized'] else "UNAUTHORIZED"
        confidence_str = f"{result['confidence']:.2f}"
        
        self.plate_log_tree.insert('', 0, values=(time_str, result['plate_number'],
                                                 result['vehicle_type'], status, confidence_str))
        
        children = self.plate_log_tree.get_children()
        if len(children) > 100:
            self.plate_log_tree.delete(children[-1])
    
    def add_to_fire_log(self, result):
        """Add fire detection result to log"""
        time_str = result['timestamp'].strftime("%H:%M:%S")
        class_name = result['class'].upper()
        confidence_str = f"{result['confidence']:.2f}"
        severity = result['severity'].upper()
        alert_status = "🚨 YES" if result.get('alert', False) else "No"
        
        # Color code based on severity
        item = self.fire_log_tree.insert('', 0, values=(time_str, class_name, 
                                                        confidence_str, severity, alert_status))
        
        # Update statistics
        self._update_fire_stats()
        
        # Keep only last 100 entries
        children = self.fire_log_tree.get_children()
        if len(children) > 100:
            self.fire_log_tree.delete(children[-1])
    
    def _update_fire_stats(self):
        """Update fire detection statistics"""
        children = self.fire_log_tree.get_children()
        total = len(children)
        
        fire_count = 0
        smoke_count = 0
        critical_count = 0
        
        for child in children:
            values = self.fire_log_tree.item(child)['values']
            if len(values) >= 4:
                detection_type = values[1]
                severity = values[3]
                
                if 'FIRE' in detection_type:
                    fire_count += 1
                elif 'SMOKE' in detection_type:
                    smoke_count += 1
                
                if severity == 'CRITICAL':
                    critical_count += 1
        
        self.fire_stats_labels['total_detections'].config(text=str(total))
        self.fire_stats_labels['fire_detections'].config(text=str(fire_count))
        self.fire_stats_labels['smoke_detections'].config(text=str(smoke_count))
        self.fire_stats_labels['critical_alerts'].config(text=str(critical_count))
            
    def update_statistics(self):
        """Update system statistics display"""
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
        root = tk.Tk()
        app = IntegratedSecurityGUI(root, db_config)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()