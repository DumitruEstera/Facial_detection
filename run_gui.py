#!/usr/bin/env python3
"""
Launcher script that automatically handles OpenMP conflicts
Save as: run_gui.py
"""

import os
import sys
import subprocess

def main():
    # Set environment variable
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    print("Starting Security System with OpenMP fix...")
    print("This launcher prevents TensorFlow/PyTorch conflicts")
    print("-" * 50)
    
    # Check which GUI to run
    if len(sys.argv) > 1:
        if sys.argv[1] == "integrated":
            script = "integrated_gui.py"
        elif sys.argv[1] == "face":
            script = "gui_application.py"
        else:
            script = sys.argv[1]
    else:
        # Default to face recognition GUI
        script = "gui_application.py"
        print(f"Running default: {script}")
        print("Use 'python run_gui.py integrated' for integrated GUI")
    
    # Run the GUI
    try:
        subprocess.run([sys.executable, script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: {script} not found!")
        print("\nAvailable options:")
        print("  python run_gui.py                # Face recognition GUI")
        print("  python run_gui.py integrated     # Integrated GUI (face + plates)")
        print("  python run_gui.py your_script.py # Custom script")
        sys.exit(1)

if __name__ == "__main__":
    main()