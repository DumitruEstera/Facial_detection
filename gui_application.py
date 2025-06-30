import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import queue
from datetime import datetime
from facial_recognition_system import FacialRecognitionSystem

class FacialRecognitionGUI:
    def __init__(self, root, db_config):
        self.root = root
        self.root.title("Facial Recognition Security System")
        self.root.geometry("1200x700")
        
        # Initialize system
        self.system = FacialRecognitionSystem(db_config)
        
        # Video capture
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Setup GUI
        self.setup_gui()
        
        # Start video thread
        self.video_thread = None
        
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel - Video feed
        video_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding="10")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.grid(row=0, column=0)
        
        # Control buttons
        control_frame = ttk.Frame(video_frame)
        control_frame.grid(row=1, column=0, pady=10)
        
        self.start_btn = ttk.Button(control_frame, text="Start Camera", 
                                   command=self.start_camera)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Camera", 
                                  command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        # Right panel - Information and controls
        info_frame = ttk.LabelFrame(main_frame, text="Information", padding="10")
        info_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Recognition log
        log_label = ttk.Label(info_frame, text="Recognition Log:")
        log_label.grid(row=0, column=0, sticky=tk.W)
        
        # Create treeview for log
        self.log_tree = ttk.Treeview(info_frame, columns=('Time', 'Name', 'ID', 'Confidence'), 
                                    show='tree headings', height=10)
        self.log_tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Configure columns
        self.log_tree.column('#0', width=0, stretch=tk.NO)
        self.log_tree.column('Time', width=150)
        self.log_tree.column('Name', width=150)
        self.log_tree.column('ID', width=100)
        self.log_tree.column('Confidence', width=100)
        
        self.log_tree.heading('Time', text='Time')
        self.log_tree.heading('Name', text='Name')
        self.log_tree.heading('ID', text='Employee ID')
        self.log_tree.heading('Confidence', text='Confidence')
        
        # Scrollbar for log
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.log_tree.yview)
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.log_tree.configure(yscrollcommand=scrollbar.set)
        
        # Registration section
        reg_frame = ttk.LabelFrame(info_frame, text="Registration", padding="10")
        reg_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Registration fields
        ttk.Label(reg_frame, text="Name:").grid(row=0, column=0, sticky=tk.W)
        self.name_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.name_var, width=25).grid(row=0, column=1, padx=5)
        
        ttk.Label(reg_frame, text="Employee ID:").grid(row=1, column=0, sticky=tk.W)
        self.emp_id_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.emp_id_var, width=25).grid(row=1, column=1, padx=5)
        
        ttk.Label(reg_frame, text="Department:").grid(row=2, column=0, sticky=tk.W)
        self.dept_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.dept_var, width=25).grid(row=2, column=1, padx=5)
        
        # Registration button
        self.register_btn = ttk.Button(reg_frame, text="Start Registration", 
                                      command=self.start_registration)
        self.register_btn.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Statistics section
        stats_frame = ttk.LabelFrame(info_frame, text="System Statistics", padding="10")
        stats_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=10)
        
        self.stats_label = ttk.Label(stats_frame, text="")
        self.stats_label.grid(row=0, column=0)
        
        # Update statistics
        self.update_statistics()
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
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
        """Process video frames in a separate thread"""
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Process frame for face recognition
                    annotated_frame, results = self.system.process_frame(frame)
                    
                    # Add to queue
                    try:
                        self.frame_queue.put((annotated_frame, results), block=False)
                    except queue.Full:
                        pass
                        
    def update_display(self):
        """Update the video display"""
        try:
            frame, results = self.frame_queue.get(block=False)
            
            # Convert frame to RGB and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (640, 480))
            
            # Convert to PIL Image
            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image=image)
            
            # Update label
            self.video_label.config(image=photo)
            self.video_label.image = photo
            
            # Update log if there are results
            for result in results:
                self.add_to_log(result)
                
        except queue.Empty:
            pass
            
        if self.is_running:
            self.root.after(30, self.update_display)
            
    def add_to_log(self, result):
        """Add recognition result to log"""
        time_str = result['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        confidence_str = f"{result['confidence']:.2f}"
        
        # Insert at the beginning
        self.log_tree.insert('', 0, values=(time_str, result['name'], 
                                           result['employee_id'], confidence_str))
        
        # Keep only last 100 entries
        children = self.log_tree.get_children()
        if len(children) > 100:
            self.log_tree.delete(children[-1])
            
    def start_registration(self):
        """Start the registration process"""
        name = self.name_var.get().strip()
        emp_id = self.emp_id_var.get().strip()
        dept = self.dept_var.get().strip()
        
        if not name or not emp_id:
            messagebox.showerror("Error", "Name and Employee ID are required")
            return
            
        # Create registration window
        reg_window = tk.Toplevel(self.root)
        reg_window.title(f"Register {name}")
        reg_window.geometry("800x600")
        
        # Registration interface
        reg_label = ttk.Label(reg_window, text="Position your face in the camera and press 'C' to capture")
        reg_label.pack(pady=10)
        
        video_label = ttk.Label(reg_window)
        video_label.pack()
        
        captured_label = ttk.Label(reg_window, text="Captured: 0/10")
        captured_label.pack(pady=5)
        
        # Capture variables
        captured_faces = []
        
        def capture_registration_frames():
            cap = cv2.VideoCapture(0)
            
            while len(captured_faces) < 10 and reg_window.winfo_exists():
                ret, frame = cap.read()
                if ret:
                    # Detect faces
                    faces = self.system.face_detector.detect_faces(frame)
                    
                    # Draw faces
                    display_frame = self.system.face_detector.draw_faces(frame, faces)
                    
                    # Convert and display
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (640, 480))
                    image = Image.fromarray(frame_resized)
                    photo = ImageTk.PhotoImage(image=image)
                    video_label.config(image=photo)
                    video_label.image = photo
                    
                    # Check for capture
                    if cv2.waitKey(1) & 0xFF == ord('c') and len(faces) == 1:
                        face_image = self.system.face_detector.extract_face(frame, faces[0])
                        if face_image is not None:
                            captured_faces.append(face_image)
                            captured_label.config(text=f"Captured: {len(captured_faces)}/10")
                            
            cap.release()
            
            if len(captured_faces) == 10:
                # Register person
                success = self.system.register_person(
                    name=name,
                    employee_id=emp_id,
                    face_images=captured_faces,
                    department=dept if dept else None
                )
                
                if success:
                    messagebox.showinfo("Success", f"Successfully registered {name}")
                    self.update_statistics()
                else:
                    messagebox.showerror("Error", "Registration failed")
                    
            reg_window.destroy()
            
        # Start capture thread
        capture_thread = threading.Thread(target=capture_registration_frames)
        capture_thread.daemon = True
        capture_thread.start()
        
    def update_statistics(self):
        """Update system statistics display"""
        stats = self.system.faiss_index.get_statistics()
        stats_text = f"Total Embeddings: {stats['total_embeddings']}\n"
        stats_text += f"Registered Persons: {stats['unique_persons']}\n"
        stats_text += f"Index Type: {stats['index_type']}"
        self.stats_label.config(text=stats_text)
        
    def on_closing(self):
        """Handle window closing"""
        self.stop_camera()
        self.system.cleanup()
        self.root.destroy()


def main():
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'facial_recognition_db',
        'user': 'postgres',
        'password': 'your_password'  # Change this
    }
    
    # Create GUI
    root = tk.Tk()
    app = FacialRecognitionGUI(root, db_config)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start GUI
    root.mainloop()


if __name__ == "__main__":
    main()