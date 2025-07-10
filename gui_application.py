import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import queue
from datetime import datetime
from facial_recognition_system import FacialRecognitionSystem
from license_plate_recognition_system import LicensePlateRecognitionSystem

class IntegratedSecurityGUI:
    def __init__(self, root, db_config):
        self.root = root
        self.root.title("Integrated Security System - Face & License Plate Recognition")
        self.root.geometry("1400x800")
        
        # Initialize systems
        self.face_system = FacialRecognitionSystem(db_config, camera_id="0")
        self.plate_system = LicensePlateRecognitionSystem(db_config)
        
        # Video capture
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Current mode
        self.mode = tk.StringVar(value="both")  # "face", "plate", or "both"
        
        # Setup GUI
        self.setup_gui()
        
        # Start video thread
        self.video_thread = None
        
    def setup_gui(self):
        """Setup the GUI layout"""
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
        
        # Camera controls
        self.start_btn = ttk.Button(control_panel, text="Start Camera", 
                                   command=self.start_camera)
        self.start_btn.grid(row=0, column=4, padx=20)
        
        self.stop_btn = ttk.Button(control_panel, text="Stop Camera", 
                                  command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=5, padx=5)
        
        # Left panel - Video feed
        video_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding="10")
        video_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.grid(row=0, column=0)
        
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
        # Face recognition log
        log_label = ttk.Label(self.face_tab, text="Face Recognition Log:")
        log_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        
        # Create treeview for log
        self.face_log_tree = ttk.Treeview(self.face_tab, 
                                         columns=('Time', 'Name', 'ID', 'Confidence'), 
                                         show='tree headings', height=10)
        self.face_log_tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)
        
        # Configure columns
        self.face_log_tree.column('#0', width=0, stretch=tk.NO)
        self.face_log_tree.column('Time', width=150)
        self.face_log_tree.column('Name', width=150)
        self.face_log_tree.column('ID', width=100)
        self.face_log_tree.column('Confidence', width=100)
        
        self.face_log_tree.heading('Time', text='Time')
        self.face_log_tree.heading('Name', text='Name')
        self.face_log_tree.heading('ID', text='Employee ID')
        self.face_log_tree.heading('Confidence', text='Confidence')
        
        # Scrollbar
        face_scrollbar = ttk.Scrollbar(self.face_tab, orient=tk.VERTICAL, 
                                      command=self.face_log_tree.yview)
        face_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.face_log_tree.configure(yscrollcommand=face_scrollbar.set)
        
        # Face registration section
        reg_frame = ttk.LabelFrame(self.face_tab, text="Face Registration", padding="10")
        reg_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        
        ttk.Label(reg_frame, text="Name:").grid(row=0, column=0, sticky=tk.W)
        self.face_name_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.face_name_var, width=25).grid(row=0, column=1, padx=5)
        
        ttk.Label(reg_frame, text="Employee ID:").grid(row=1, column=0, sticky=tk.W)
        self.face_emp_id_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.face_emp_id_var, width=25).grid(row=1, column=1, padx=5)
        
        ttk.Label(reg_frame, text="Department:").grid(row=2, column=0, sticky=tk.W)
        self.face_dept_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.face_dept_var, width=25).grid(row=2, column=1, padx=5)
        
        ttk.Button(reg_frame, text="Register Face", 
                  command=self.register_face).grid(row=3, column=0, columnspan=2, pady=10)
        
    def setup_plate_tab(self):
        """Setup license plate tab"""
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
        
        ttk.Label(reg_frame, text="Owner ID:").grid(row=2, column=0, sticky=tk.W)
        self.plate_owner_id_var = tk.StringVar()
        ttk.Entry(reg_frame, textvariable=self.plate_owner_id_var, width=25).grid(row=2, column=1, padx=5)
        
        ttk.Label(reg_frame, text="Vehicle Type:").grid(row=3, column=0, sticky=tk.W)
        self.plate_vehicle_var = tk.StringVar()
        vehicle_combo = ttk.Combobox(reg_frame, textvariable=self.plate_vehicle_var, 
                                    values=['car', 'motorcycle', 'truck', 'bus'], width=22)
        vehicle_combo.grid(row=3, column=1, padx=5)
        
        self.plate_authorized_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(reg_frame, text="Authorized", 
                       variable=self.plate_authorized_var).grid(row=4, column=0, columnspan=2, pady=5)
        
        ttk.Button(reg_frame, text="Register Plate", 
                  command=self.register_plate).grid(row=5, column=0, columnspan=2, pady=10)
        
    def setup_stats_tab(self):
        """Setup statistics tab"""
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
        """Process video frames in a separate thread"""
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    mode = self.mode.get()
                    face_results = []
                    plate_results = []
                    annotated_frame = frame.copy()
                    
                    # Process based on mode
                    if mode in ['face', 'both']:
                        face_frame, face_results = self.face_system.process_frame(frame)
                        if mode == 'face':
                            annotated_frame = face_frame
                            
                    if mode in ['plate', 'both']:
                        plate_results = self.plate_system.process_frame(frame)
                        plate_frame = self.plate_system.draw_outputs(frame.copy(), plate_results)
                        if mode == 'plate':
                            annotated_frame = plate_frame
                            
                    if mode == 'both':
                        annotated_frame = plate_frame
                        
                    # Add to queue
                    try:
                        self.frame_queue.put((annotated_frame, face_results, plate_results), 
                                           block=False)
                    except queue.Full:
                        pass
                        
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
                self.add_to_face_log(result)
                
            for result in plate_results:
                self.add_to_plate_log(result)
                
        except queue.Empty:
            pass
            
        if self.is_running:
            self.root.after(30, self.update_display)
            
    def add_to_face_log(self, result):
        """Add face recognition result to log"""
        time_str = result['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        confidence_str = f"{result['confidence']:.2f}"
        
        self.face_log_tree.insert('', 0, values=(time_str, result['name'], 
                                                result['employee_id'], confidence_str))
        
        # Keep only last 100 entries
        children = self.face_log_tree.get_children()
        if len(children) > 100:
            self.face_log_tree.delete(children[-1])
            
    def add_to_plate_log(self, result):
        """Add license plate result to log"""
        time_str = result['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        status = "AUTHORIZED" if result['is_authorized'] else "UNAUTHORIZED"
        confidence_str = f"{result['confidence']:.2f}"
        
        self.plate_log_tree.insert('', 0, values=(time_str, result['plate_number'],
                                                 result['vehicle_type'], status, confidence_str))
        
        # Keep only last 100 entries
        children = self.plate_log_tree.get_children()
        if len(children) > 100:
            self.plate_log_tree.delete(children[-1])
            
    def register_face(self):
        """Handle face registration"""
        name = self.face_name_var.get().strip()
        emp_id = self.face_emp_id_var.get().strip()
        dept = self.face_dept_var.get().strip()
        
        if not name or not emp_id:
            messagebox.showerror("Error", "Name and Employee ID are required")
            return
            
        # Stop camera temporarily
        was_running = self.is_running
        if was_running:
            self.stop_camera()
            
        messagebox.showinfo("Info", "Please use the example_usage.py script in registration mode for face registration")
        
        # Restart camera if it was running
        if was_running:
            self.start_camera()
            
    def register_plate(self):
        """Handle license plate registration"""
        plate_number = self.plate_number_var.get().strip()
        owner_name = self.plate_owner_var.get().strip()
        owner_id = self.plate_owner_id_var.get().strip()
        vehicle_type = self.plate_vehicle_var.get()
        is_authorized = self.plate_authorized_var.get()
        
        if not plate_number:
            messagebox.showerror("Error", "Plate number is required")
            return
            
        try:
            success = self.plate_system.register_plate(
                plate_number=plate_number,
                owner_name=owner_name,
                owner_id=owner_id,
                vehicle_type=vehicle_type,
                is_authorized=is_authorized
            )
            
            if success:
                messagebox.showinfo("Success", f"Successfully registered plate: {plate_number}")
                # Clear fields
                self.plate_number_var.set("")
                self.plate_owner_var.set("")
                self.plate_owner_id_var.set("")
                self.update_statistics()
            else:
                messagebox.showerror("Error", "Failed to register plate")
                
        except Exception as e:
            messagebox.showerror("Error", f"Registration failed: {str(e)}")
            
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
    
    # Create GUI
    root = tk.Tk()
    app = IntegratedSecurityGUI(root, db_config)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start GUI
    root.mainloop()


if __name__ == "__main__":
    main()