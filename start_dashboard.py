#!/usr/bin/env python3
"""
DriveToQuizlet Dashboard Startup Script

This script starts the web dashboard for the DriveToQuizlet application.
It handles environment setup, database initialization, and server startup.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['fastapi', 'uvicorn', 'jinja2']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        
        # Try to install missing packages
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                'fastapi', 'uvicorn[standard]', 'jinja2', 'python-multipart'
            ])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please install manually:")
            print("   pip install fastapi uvicorn[standard] jinja2 python-multipart")
            return False
    
    return True

def setup_environment():
    """Set up the environment and required directories."""
    # Ensure required directories exist
    required_dirs = [
        'config',
        'static/css',
        'static/js', 
        'static/images',
        'templates',
        'lessons'
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure verified")

def start_server(host='localhost', port=8000, reload=True):
    """Start the FastAPI server."""
    print(f"🚀 Starting DriveToQuizlet Dashboard...")
    print(f"📱 Dashboard will be available at: http://{host}:{port}")
    print(f"🔄 Auto-reload: {'enabled' if reload else 'disabled'}")
    print("-" * 50)
    
    try:
        # Import and run the server
        import uvicorn
        uvicorn.run(
            "scripts.web_dashboard:app",
            host=host,
            port=port,
            reload=reload,
            reload_dirs=["scripts", "templates", "static"],
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("💡 Make sure you're in the correct directory and all files are present")

def main():
    """Main startup function."""
    print("🧠 DriveToQuizlet Dashboard Startup")
    print("=" * 40)
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Check and install dependencies
    if not check_dependencies():
        return 1
    
    # Setup environment
    setup_environment()
    
    # Start the server
    start_server()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
