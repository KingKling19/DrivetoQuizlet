import sys
import os
import time
import threading
import json
import subprocess
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
import webbrowser

# Try to import system tray functionality
try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_PYSTRAY = True
except ImportError:
    HAS_PYSTRAY = False
    print("pystray not available - system tray functionality disabled")

class DesktopMonitor:
    def __init__(self):
        self.config = self.load_config()
        self.monitor_thread = None
        self.running = False
        self.web_server_process = None
        
    def load_config(self):
        """Load configuration from config file"""
        config_path = Path("config/drive_config.json")
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        else:
            return {
                "web_server": {"host": "localhost", "port": 8000},
                "processing": {"check_interval_hours": 1}
            }
    
    def create_tray_icon(self):
        """Create system tray icon"""
        if not HAS_PYSTRAY:
            return None
        
        # Create a simple icon
        image = Image.new('RGB', (64, 64), color='purple')
        dc = ImageDraw.Draw(image)
        dc.rectangle([16, 16, 48, 48], fill='white')
        
        # Create menu
        menu = pystray.Menu(
            pystray.MenuItem("Open Dashboard", self.open_dashboard),
            pystray.MenuItem("Scan Drive Now", self.scan_drive),
            pystray.MenuItem("Start Web Server", self.start_web_server),
            pystray.MenuItem("Stop Web Server", self.stop_web_server),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self.quit_app)
        )
        
        return pystray.Icon("DriveToQuizlet", image, "DriveToQuizlet Monitor", menu)
    
    def open_dashboard(self, icon=None, item=None):
        """Open the web dashboard"""
        url = f"http://{self.config['web_server']['host']}:{self.config['web_server']['port']}"
        webbrowser.open(url)
    
    def scan_drive(self, icon=None, item=None):
        """Manually trigger a drive scan"""
        try:
            # Import and run the drive monitor scan
            from src.drive.drive_monitor import DriveMonitor
            monitor = DriveMonitor()
            new_files = monitor.scan_all_folders()
            
            if new_files:
                messagebox.showinfo("Drive Scan", f"Found {len(new_files)} new files!")
            else:
                messagebox.showinfo("Drive Scan", "No new files found.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error scanning drive: {str(e)}")
    
    def start_web_server(self, icon=None, item=None):
        """Start the web server"""
        if self.web_server_process and self.web_server_process.poll() is None:
            messagebox.showinfo("Web Server", "Web server is already running!")
            return
        
        try:
            self.web_server_process = subprocess.Popen([
                sys.executable, "scripts/web_dashboard.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a moment for server to start
            time.sleep(2)
            
            if self.web_server_process.poll() is None:
                messagebox.showinfo("Web Server", "Web server started successfully!")
                self.open_dashboard()
            else:
                messagebox.showerror("Error", "Failed to start web server")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error starting web server: {str(e)}")
    
    def stop_web_server(self, icon=None, item=None):
        """Stop the web server"""
        if self.web_server_process and self.web_server_process.poll() is None:
            self.web_server_process.terminate()
            messagebox.showinfo("Web Server", "Web server stopped!")
        else:
            messagebox.showinfo("Web Server", "Web server is not running!")
    
    def quit_app(self, icon=None, item=None):
        """Quit the application"""
        self.running = False
        if self.web_server_process and self.web_server_process.poll() is None:
            self.web_server_process.terminate()
        
        if icon:
            icon.stop()
        else:
            sys.exit(0)
    
    def run_monitor_loop(self):
        """Run the monitoring loop"""
        from src.drive.drive_monitor import DriveMonitor
        monitor = DriveMonitor()
        
        while self.running:
            try:
                print(f"[{datetime.now()}] Running scheduled drive scan...")
                new_files = monitor.scan_all_folders()
                
                if new_files:
                    print(f"Found {len(new_files)} new files")
                    # Could add desktop notification here
                
            except Exception as e:
                print(f"Error in monitor loop: {str(e)}")
            
            # Sleep for the configured interval
            time.sleep(self.config['processing']['check_interval_hours'] * 3600)
    
    def start_monitoring(self):
        """Start the desktop monitor"""
        self.running = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.run_monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # Create and run system tray icon
        if HAS_PYSTRAY:
            icon = self.create_tray_icon()
            if icon:
                print("Starting system tray monitor...")
                icon.run()
        else:
            # Fallback to console mode
            print("Starting console monitor...")
            print("Press Ctrl+C to stop")
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.quit_app()

def main():
    """Main entry point"""
    print("DriveToQuizlet Desktop Monitor")
    print("=" * 40)
    
    # Check if required files exist
    required_files = [
        "config/drive_config.json",
        "config/token.json",
        "scripts/drive_monitor.py",
        "scripts/web_dashboard.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("Error: Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease ensure all required files are present before running the monitor.")
        return
    
    # Start the monitor
    monitor = DesktopMonitor()
    monitor.start_monitoring()

if __name__ == "__main__":
    main()

