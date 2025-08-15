#!/usr/bin/env python3
"""
Test script to demonstrate automatic lesson creation for unknown files.
"""

from drive_automation import DriveAutomation
from pathlib import Path

def test_lesson_detection():
    """Test the lesson name detection with various filenames"""
    automation = DriveAutomation()
    
    # Test filenames that should create new lessons
    test_files = [
        "Advanced_Combat_Tactics.pptx",
        "Military_Leadership_Principles.pdf",
        "Field_Operations_Manual.jpg",
        "Strategic_Planning_Workshop.m4a",
        "Unknown_Training_Material.pptx"
    ]
    
    print("🧪 Testing lesson name detection...")
    print("=" * 50)
    
    for filename in test_files:
        lesson_name = automation.detect_lesson_name(filename)
        print(f"📄 {filename}")
        print(f"   📁 → {lesson_name}")
        print()
    
    print("✅ Lesson detection test complete!")

def test_lesson_structure_creation():
    """Test creating lesson folder structure"""
    automation = DriveAutomation()
    
    test_lesson = "Test_New_Lesson"
    
    print(f"🧪 Testing lesson structure creation for: {test_lesson}")
    print("=" * 50)
    
    # Create the lesson structure
    lesson_dir = automation.create_lesson_structure(test_lesson)
    
    # Check what was created
    if lesson_dir.exists():
        print(f"✅ Lesson directory created: {lesson_dir}")
        
        # List all subdirectories
        subdirs = [d for d in lesson_dir.iterdir() if d.is_dir()]
        print(f"📁 Subdirectories created:")
        for subdir in sorted(subdirs):
            print(f"   - {subdir.name}/")
        
        # Check for README
        readme_path = lesson_dir / "README.md"
        if readme_path.exists():
            print(f"📝 README.md created")
        else:
            print(f"❌ README.md not found")
    else:
        print(f"❌ Failed to create lesson directory")
    
    print("\n✅ Lesson structure test complete!")

if __name__ == "__main__":
    print("🚀 Testing automatic lesson creation functionality")
    print("=" * 60)
    
    test_lesson_detection()
    print()
    test_lesson_structure_creation()
    
    print("\n🎉 All tests complete!")


