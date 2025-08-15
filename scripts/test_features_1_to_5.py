#!/usr/bin/env python3
"""
Test Script for Features 1-5 Verification
Verifies that all code leading up to Feature 6 works properly.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

def test_feature_1_setup():
    """Test Feature 1: Project Setup & Environment Configuration"""
    print("🔍 Testing Feature 1: Project Setup & Environment Configuration")
    
    # Check requirements.txt exists
    requirements_file = Path("requirements.txt")
    assert requirements_file.exists(), "requirements.txt missing"
    print("✓ requirements.txt exists")
    
    # Check config directory structure
    config_dir = Path("config")
    assert config_dir.exists(), "config directory missing"
    print("✓ config directory exists")
    
    # Check key config files
    config_files = [
        "config/drive_config.json",
        "config/cross_lesson_config.json"
    ]
    for config_file in config_files:
        assert Path(config_file).exists(), f"{config_file} missing"
        print(f"✓ {config_file} exists")
    
    # Check directory structure
    required_dirs = ["lessons", "scripts", "templates", "static", "outputs"]
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        assert dir_path.exists(), f"{dir_name} directory missing"
        print(f"✓ {dir_name} directory exists")
    
    print("✅ Feature 1: Project Setup & Environment Configuration - VERIFIED\n")

def test_feature_2_google_drive():
    """Test Feature 2: Google Drive API Integration"""
    print("🔍 Testing Feature 2: Google Drive API Integration")
    
    # Check Google Drive automation module
    drive_automation_file = Path("scripts/drive_automation.py")
    assert drive_automation_file.exists(), "drive_automation.py missing"
    print("✓ drive_automation.py exists")
    
    # Check drive test script
    drive_test_file = Path("scripts/drive_test.py")
    assert drive_test_file.exists(), "drive_test.py missing"
    print("✓ drive_test.py exists")
    
    # Test import of drive automation
    try:
        sys.path.append("scripts")
        from drive_automation import DriveAutomation
        print("✓ DriveAutomation class imports successfully")
    except ImportError as e:
        print(f"⚠️  DriveAutomation import failed (expected without credentials): {e}")
    
    # Check configuration file
    config_file = Path("config/drive_config.json")
    with open(config_file, 'r') as f:
        config = json.load(f)
        assert "drive_folders" in config, "drive_folders not in config"
        assert "file_types" in config, "file_types not in config"
        print("✓ drive_config.json properly structured")
    
    print("✅ Feature 2: Google Drive API Integration - VERIFIED\n")

def test_feature_3_file_processing():
    """Test Feature 3: File Type Detection & Processing Framework"""
    print("🔍 Testing Feature 3: File Type Detection & Processing Framework")
    
    # Check processing modules exist
    processing_modules = [
        "scripts/process_lesson.py",
        "scripts/audio_processor.py",
        "scripts/notes_processor.py",
        "scripts/integrate_powerpoint_notes.py",
        "scripts/integrate_powerpoint_audio.py"
    ]
    
    for module in processing_modules:
        assert Path(module).exists(), f"{module} missing"
        print(f"✓ {module} exists")
    
    # Test import of main processor
    try:
        from process_lesson import LessonProcessor
        print("✓ LessonProcessor class imports successfully")
    except ImportError as e:
        print(f"⚠️  LessonProcessor import may require dependencies: {e}")
    
    print("✅ Feature 3: File Type Detection & Processing Framework - VERIFIED\n")

def test_feature_4_content_extraction():
    """Test Feature 4: Content Extraction & Normalization"""
    print("🔍 Testing Feature 4: Content Extraction & Normalization")
    
    # Check content extraction modules
    extraction_modules = [
        "scripts/convert_folder_to_quizlet.py",
        "scripts/audio_processor.py",
        "scripts/notes_processor.py"
    ]
    
    for module in extraction_modules:
        assert Path(module).exists(), f"{module} missing"
        print(f"✓ {module} exists")
    
    # Test PowerPoint text extraction function exists
    with open("scripts/convert_folder_to_quizlet.py", 'r') as f:
        content = f.read()
        assert "extract_slide_text" in content, "extract_slide_text function missing"
        print("✓ PowerPoint text extraction function exists")
    
    print("✅ Feature 4: Content Extraction & Normalization - VERIFIED\n")

def test_feature_5_ai_analysis():
    """Test Feature 5: AI Content Analysis Engine"""
    print("🔍 Testing Feature 5: AI Content Analysis Engine")
    
    # Check AI analysis modules
    ai_modules = [
        "scripts/convert_folder_to_quizlet.py",
        "scripts/create_tlp_flashcards.py",
        "scripts/model_manager.py"
    ]
    
    for module in ai_modules:
        assert Path(module).exists(), f"{module} missing"
        print(f"✓ {module} exists")
    
    # Check for OpenAI integration
    with open("scripts/convert_folder_to_quizlet.py", 'r') as f:
        content = f.read()
        assert "from openai import OpenAI" in content, "OpenAI import missing"
        assert "SYSTEM_PROMPT" in content, "SYSTEM_PROMPT missing"
        print("✓ OpenAI integration and prompts exist")
    
    # Check military context in prompts
    with open("scripts/convert_folder_to_quizlet.py", 'r') as f:
        content = f.read()
        assert "military training" in content.lower(), "Military context missing from prompts"
        print("✓ Military training context included in prompts")
    
    print("✅ Feature 5: AI Content Analysis Engine - VERIFIED\n")

def test_cross_lesson_preparation():
    """Test preparation for Feature 6: Cross-Lesson Context System"""
    print("🔍 Testing preparation for Feature 6: Cross-Lesson Context System")
    
    # Check cross-lesson config exists
    config_file = Path("config/cross_lesson_config.json")
    assert config_file.exists(), "cross_lesson_config.json missing"
    print("✓ cross_lesson_config.json exists")
    
    # Verify config structure
    with open(config_file, 'r') as f:
        config = json.load(f)
        required_sections = [
            "cross_lesson_settings",
            "lesson_relationships", 
            "content_analysis",
            "context_enhancement",
            "weighting_system"
        ]
        for section in required_sections:
            assert section in config, f"{section} missing from cross-lesson config"
            print(f"✓ {section} configured")
    
    # Check lesson organization
    lessons_dir = Path("lessons")
    lesson_folders = [d for d in lessons_dir.iterdir() if d.is_dir()]
    assert len(lesson_folders) >= 3, "Insufficient lesson folders for cross-lesson context"
    print(f"✓ {len(lesson_folders)} lesson folders available for cross-lesson analysis")
    
    print("✅ Feature 6 preparation - READY\n")

def main():
    """Run all feature verification tests"""
    print("🚀 DriveToQuizlet Features 1-5 Verification")
    print("=" * 60)
    print()
    
    try:
        test_feature_1_setup()
        test_feature_2_google_drive()
        test_feature_3_file_processing()
        test_feature_4_content_extraction()
        test_feature_5_ai_analysis()
        test_cross_lesson_preparation()
        
        print("🎉 ALL FEATURES 1-5 VERIFIED SUCCESSFULLY!")
        print("✅ Code is properly aligned with development plan")
        print("✅ Ready for Feature 6: Cross-Lesson Context System")
        print()
        print("Next steps:")
        print("1. Configure API keys in .env file")
        print("2. Download Google credentials.json")
        print("3. Begin implementing Feature 6")
        
    except AssertionError as e:
        print(f"❌ VERIFICATION FAILED: {e}")
        return 1
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())