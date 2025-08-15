#!/usr/bin/env python3
"""
Setup Validation Script for Army ADA BOLC Flashcard App
Run this script to validate your setup and identify any issues.
"""

import os
import sys
from pathlib import Path
import json

def print_status(message, status="INFO", details=None):
    """Print a status message with appropriate formatting"""
    symbols = {
        "OK": "✅",
        "WARNING": "⚠️ ",
        "ERROR": "❌",
        "INFO": "ℹ️ "
    }
    print(f"{symbols.get(status, '•')} {message}")
    if details:
        for detail in details if isinstance(details, list) else [details]:
            print(f"   {detail}")

def check_file_structure():
    """Check if the required file structure exists"""
    print("\n🏗️  Checking File Structure")
    
    required_dirs = [
        "scripts", "config", "lessons", "outputs", 
        "logs", "temp", "static", "templates"
    ]
    
    required_files = [
        "requirements.txt", ".env.template", "setup.py",
        "scripts/convert_folder_to_quizlet.py",
        "scripts/lesson_content_indexer.py",
        "scripts/cross_lesson_analyzer.py",
        "scripts/context_optimizer.py",
        "config/drive_config.json"
    ]
    
    all_good = True
    
    for directory in required_dirs:
        if Path(directory).exists():
            print_status(f"Directory '{directory}' exists", "OK")
        else:
            print_status(f"Directory '{directory}' missing", "ERROR")
            all_good = False
    
    for file_path in required_files:
        if Path(file_path).exists():
            print_status(f"File '{file_path}' exists", "OK")
        else:
            print_status(f"File '{file_path}' missing", "ERROR")
            all_good = False
    
    return all_good

def check_python_version():
    """Check Python version compatibility"""
    print("\n🐍 Checking Python Version")
    
    version = sys.version_info
    if version >= (3, 8):
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Compatible", "OK")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+", "ERROR")
        return False

def check_dependencies():
    """Check if dependencies can be imported"""
    print("\n📦 Checking Dependencies")
    
    core_deps = [
        ("json", "Built-in JSON support"),
        ("pathlib", "Built-in path handling"),
        ("sqlite3", "Built-in database support")
    ]
    
    optional_deps = [
        ("openai", "OpenAI API integration"),
        ("google.oauth2.credentials", "Google Drive API"),
        ("pptx", "PowerPoint processing (python-pptx)"),
        ("PIL", "Image processing (Pillow)"),
        ("easyocr", "OCR functionality"),
        ("fastapi", "Web dashboard"),
        ("whisper", "Audio transcription"),
        ("sklearn", "Machine learning features"),
        ("numpy", "Numerical operations"),
        ("psutil", "System monitoring")
    ]
    
    core_available = 0
    for module, description in core_deps:
        try:
            __import__(module)
            print_status(f"{description}", "OK")
            core_available += 1
        except ImportError:
            print_status(f"{description}", "ERROR", f"Cannot import {module}")
    
    optional_available = 0
    for module, description in optional_deps:
        try:
            __import__(module)
            print_status(f"{description}", "OK")
            optional_available += 1
        except ImportError:
            print_status(f"{description}", "WARNING", f"Optional: {module} not installed")
    
    print_status(f"Core dependencies: {core_available}/{len(core_deps)} available")
    print_status(f"Optional dependencies: {optional_available}/{len(optional_deps)} available")
    
    return core_available == len(core_deps)

def check_configuration():
    """Check configuration files"""
    print("\n⚙️  Checking Configuration")
    
    config_checks = []
    
    # Check .env file
    if Path(".env").exists():
        print_status(".env file exists", "OK")
        with open(".env", 'r') as f:
            env_content = f.read()
            if "OPENAI_API_KEY=your_openai_api_key_here" in env_content:
                print_status("OpenAI API key needs to be configured", "WARNING", 
                           "Edit .env file with your actual OpenAI API key")
            elif "OPENAI_API_KEY=" in env_content:
                print_status("OpenAI API key configured", "OK")
            else:
                print_status("OpenAI API key not found in .env", "WARNING")
    else:
        print_status(".env file missing", "WARNING", 
                   "Copy .env.template to .env and configure your API keys")
    
    # Check Google Drive config
    drive_config = Path("config/drive_config.json")
    if drive_config.exists():
        print_status("Google Drive config exists", "OK")
        try:
            with open(drive_config, 'r') as f:
                config_data = json.load(f)
                if config_data.get('drive_folders', {}).get('personal_notes_audio') == "REPLACE_WITH_YOUR_DRIVE_FOLDER_ID":
                    print_status("Google Drive folder IDs need configuration", "WARNING",
                               "Edit config/drive_config.json with your actual folder IDs")
                else:
                    print_status("Google Drive folder IDs configured", "OK")
        except json.JSONDecodeError:
            print_status("Google Drive config file is invalid JSON", "ERROR")
    else:
        print_status("Google Drive config missing", "ERROR")
    
    # Check for Google credentials
    if Path("config/credentials.json").exists():
        print_status("Google API credentials file exists", "OK")
    else:
        print_status("Google API credentials missing", "WARNING",
                   "Download credentials.json from Google Cloud Console")
    
    return True

def check_sample_data():
    """Check if sample data is available for testing"""
    print("\n📚 Checking Sample Data")
    
    lesson_dirs = list(Path("lessons").glob("*/"))
    if lesson_dirs:
        print_status(f"Found {len(lesson_dirs)} lesson directories", "OK")
        
        for lesson_dir in lesson_dirs:
            output_files = list(lesson_dir.glob("output/*.json"))
            if output_files:
                print_status(f"Lesson '{lesson_dir.name}' has sample data", "OK")
            else:
                print_status(f"Lesson '{lesson_dir.name}' has no sample data", "INFO")
    else:
        print_status("No lesson directories found", "WARNING",
                   "Create lesson directories or run sample data generation")
    
    return True

def run_feature_tests():
    """Test key features that don't require external dependencies"""
    print("\n🧪 Testing Core Features")
    
    # Test lesson indexer
    try:
        sys.path.append('scripts')
        from lesson_content_indexer import LessonContentIndexer
        indexer = LessonContentIndexer()
        print_status("Lesson Content Indexer can be imported", "OK")
        
        # Test basic functionality
        lessons = indexer._get_available_lessons() if hasattr(indexer, '_get_available_lessons') else []
        print_status(f"Found {len(lessons)} lessons for indexing", "INFO")
        
    except Exception as e:
        print_status("Lesson Content Indexer test failed", "ERROR", str(e))
    
    # Test cross-lesson analyzer
    try:
        from cross_lesson_analyzer import CrossLessonAnalyzer
        print_status("Cross-Lesson Analyzer can be imported", "OK")
    except Exception as e:
        print_status("Cross-Lesson Analyzer test failed", "ERROR", str(e))
    
    # Test context optimizer
    try:
        from context_optimizer import ContextOptimizer
        print_status("Context Optimizer can be imported", "OK")
    except Exception as e:
        print_status("Context Optimizer test failed", "ERROR", str(e))
    
    return True

def main():
    """Main validation function"""
    print("🎯 Army ADA BOLC Flashcard App - Setup Validation")
    print("=" * 60)
    
    all_checks = [
        ("Python Version", check_python_version),
        ("File Structure", check_file_structure),
        ("Dependencies", check_dependencies),
        ("Configuration", check_configuration),
        ("Sample Data", check_sample_data),
        ("Feature Tests", run_feature_tests)
    ]
    
    results = {}
    for check_name, check_func in all_checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print_status(f"Error during {check_name} check: {e}", "ERROR")
            results[check_name] = False
    
    # Summary
    print("\n📊 Validation Summary")
    print("-" * 30)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for check_name, result in results.items():
        status = "OK" if result else "NEEDS ATTENTION"
        print_status(f"{check_name}: {status}", "OK" if result else "WARNING")
    
    print(f"\n✅ {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 Setup validation complete! Your system is ready.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Configure API keys in .env file")
        print("3. Set up Google Drive credentials")
        print("4. Run: python scripts/lesson_content_indexer.py")
    else:
        print("\n⚠️  Some issues found. Please address the warnings and errors above.")
        print("\nRecommended actions:")
        print("1. Run: python setup.py")
        print("2. Install missing dependencies: pip install -r requirements.txt")
        print("3. Configure .env file with your API keys")
        print("4. Set up Google Drive integration")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)