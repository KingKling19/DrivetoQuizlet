#!/usr/bin/env python3
"""
DriveToQuizlet Enhanced Dashboard Launcher

A simple script to launch the enhanced dashboard with dependency checks
and helpful error messages.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'jinja2',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_configuration():
    """Check if the system is properly configured"""
    config_files = [
        "config/drive_config.json",
        "config/token.json"
    ]
    
    missing_files = []
    
    for config_file in config_files:
        if not Path(config_file).exists():
            missing_files.append(config_file)
    
    if missing_files:
        print("⚠️  Missing configuration files:")
        for config_file in missing_files:
            print(f"   - {config_file}")
        print("\n💡 Set up configuration with:")
        print("   python scripts/drive_test.py")
        print("   # Then edit config/drive_config.json")
        return False
    
    return True

def check_lessons_directory():
    """Check if lessons directory exists"""
    if not Path("lessons").exists():
        print("⚠️  Lessons directory not found")
        print("💡 Create lessons directory with:")
        print("   mkdir lessons")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🚀 DriveToQuizlet Enhanced Dashboard Launcher")
    print("=" * 50)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ All dependencies found")
    
    # Check configuration
    print("\n⚙️  Checking configuration...")
    if not check_configuration():
        print("⚠️  Configuration issues found, but continuing...")
    else:
        print("✅ Configuration looks good")
    
    # Check lessons directory
    print("\n📁 Checking lessons directory...")
    if not check_lessons_directory():
        print("⚠️  Lessons directory missing, but continuing...")
    else:
        print("✅ Lessons directory found")
    
    # Check if enhanced dashboard exists
    dashboard_script = Path("scripts/enhanced_dashboard.py")
    if not dashboard_script.exists():
        print("\n❌ Enhanced dashboard script not found!")
        print("💡 Make sure enhanced_dashboard.py exists in the scripts directory")
        sys.exit(1)
    
    print("\n🎯 Starting Enhanced Dashboard...")
    print("📱 Dashboard will be available at: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Change to the project root directory
        os.chdir(Path(__file__).parent.parent)
        
        # Start the enhanced dashboard
        subprocess.run([
            sys.executable, "scripts/enhanced_dashboard.py"
        ])
        
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
