#!/usr/bin/env python3
"""
Lesson Organization System

Creates and manages lesson-based folder structure for easy processing
of PowerPoint presentations and handwritten notes.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

def create_lesson_structure(lesson_name: str, base_dir: Path = Path("downloads")) -> Dict[str, Path]:
    """Create organized folder structure for a lesson."""
    
    # Clean lesson name for folder naming
    clean_name = lesson_name.replace(" ", "_").replace("(", "").replace(")", "")
    
    # Create lesson directories
    lesson_dir = base_dir / "lessons" / clean_name
    presentations_dir = lesson_dir / "presentations"
    notes_dir = lesson_dir / "notes"
    audio_dir = lesson_dir / "audio"
    processed_dir = lesson_dir / "processed"
    output_dir = lesson_dir / "output"
    
    # Create all directories
    for dir_path in [lesson_dir, presentations_dir, notes_dir, audio_dir, processed_dir, output_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create README for the lesson
    readme_file = lesson_dir / "README.md"
    if not readme_file.exists():
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(f"# {lesson_name}\n\n")
            f.write("## Folder Structure\n")
            f.write("- `presentations/` - PowerPoint files (.pptx)\n")
            f.write("- `notes/` - Handwritten notes images (.png, .jpg, etc.)\n")
            f.write("- `audio/` - Lecture audio files (.m4a, .mp3, .wav)\n")
            f.write("- `processed/` - AI processing results\n")
            f.write("- `output/` - Final Quizlet flashcards (.tsv)\n\n")
            f.write("## Processing Commands\n")
            f.write("```bash\n")
            f.write(f"# Process PowerPoint only\n")
            f.write(f"python convert_folder_to_quizlet.py \"{lesson_dir}/presentations\"\n\n")
            f.write(f"# Process PowerPoint + Notes\n")
            f.write(f"python integrate_powerpoint_notes.py \"{lesson_dir}/presentations/\" \"{lesson_dir}/notes/\"\n\n")
            f.write(f"# Process PowerPoint + Audio\n")
            f.write(f"python integrate_powerpoint_audio.py \"{lesson_dir}/presentations/\" \"{lesson_dir}/audio/\"\n\n")
            f.write(f"# Process all three (PowerPoint + Notes + Audio)\n")
            f.write(f"python integrate_all_sources.py \"{lesson_dir}\"\n")
            f.write("```\n")
    
    return {
        "lesson_dir": lesson_dir,
        "presentations_dir": presentations_dir,
        "notes_dir": notes_dir,
        "audio_dir": audio_dir,
        "processed_dir": processed_dir,
        "output_dir": output_dir
    }

def organize_existing_files(lesson_name: str, base_dir: Path = Path("downloads")) -> Dict[str, List[str]]:
    """Organize existing files into lesson structure."""
    
    # Create lesson structure
    dirs = create_lesson_structure(lesson_name, base_dir)
    
    # Find and move existing files
    moved_files = {"presentations": [], "notes": [], "audio": []}
    
    # Use optimized file copying for better performance
    try:
        from optimized_file_operations import copy_lesson_files_optimized
        results = copy_lesson_files_optimized(lesson_name, base_dir)
        
        # Update moved_files based on results
        for file_path in results["successful"]:
            file_name = Path(file_path).name
            if file_name.endswith('.pptx'):
                moved_files["presentations"].append(file_name)
            elif file_name.endswith(('.jpg', '.png', '.jpeg')):
                moved_files["notes"].append(file_name)
            elif file_name.endswith('.m4a'):
                moved_files["audio"].append(file_name)
        
        print(f"✓ Optimized copying completed: {len(results['successful'])} files")
        
    except ImportError:
        # Fallback to original method if optimized module not available
        print("⚠️  Using fallback copying method")
        
        # Look for PowerPoint files
        for pptx_file in base_dir.rglob("*.pptx"):
            if lesson_name.lower() in pptx_file.name.lower():
                dest = dirs["presentations_dir"] / pptx_file.name
                if not dest.exists():
                    shutil.copy2(pptx_file, dest)
                    moved_files["presentations"].append(pptx_file.name)
                    print(f"✓ Moved presentation: {pptx_file.name}")
        
        # Look for Quizlet files
        for quizlet_file in base_dir.rglob("*.quizlet.json"):
            if lesson_name.lower() in quizlet_file.name.lower():
                dest = dirs["processed_dir"] / quizlet_file.name
                if not dest.exists():
                    shutil.copy2(quizlet_file, dest)
                    print(f"✓ Moved Quizlet file: {quizlet_file.name}")
        
        # Look for note images
        for note_file in base_dir.rglob("*.jpg"):
            if lesson_name.lower() in note_file.name.lower():
                dest = dirs["notes_dir"] / note_file.name
                if not dest.exists():
                    shutil.copy2(note_file, dest)
                    moved_files["notes"].append(note_file.name)
                    print(f"✓ Moved note: {note_file.name}")
        
        for note_file in base_dir.rglob("*.png"):
            if lesson_name.lower() in note_file.name.lower():
                dest = dirs["notes_dir"] / note_file.name
                if not dest.exists():
                    shutil.copy2(note_file, dest)
                    moved_files["notes"].append(note_file.name)
                    print(f"✓ Moved note: {note_file.name}")
        
        # Look for audio files
        for audio_file in base_dir.rglob("*.m4a"):
            if lesson_name.lower() in audio_file.name.lower():
                dest = dirs["audio_dir"] / audio_file.name
                if not dest.exists():
                    shutil.copy2(audio_file, dest)
                    moved_files["audio"].append(audio_file.name)
                    print(f"✓ Moved audio: {audio_file.name}")
    
    return moved_files

def list_lessons(base_dir: Path = Path("downloads")) -> List[str]:
    """List all existing lessons."""
    lessons_dir = base_dir / "lessons"
    if not lessons_dir.exists():
        return []
    
    lessons = []
    for lesson_dir in lessons_dir.iterdir():
        if lesson_dir.is_dir():
            lessons.append(lesson_dir.name)
    
    return sorted(lessons)

def show_lesson_status(lesson_name: str, base_dir: Path = Path("downloads")) -> Dict[str, Any]:
    """Show status of files in a lesson."""
    
    dirs = create_lesson_structure(lesson_name, base_dir)
    
    status = {
        "lesson_name": lesson_name,
        "presentations": [],
        "notes": [],
        "audio": [],
        "processed": [],
        "output": []
    }
    
    # Check presentations
    for file in dirs["presentations_dir"].glob("*"):
        if file.is_file():
            status["presentations"].append(file.name)
    
    # Check notes
    for file in dirs["notes_dir"].glob("*"):
        if file.is_file():
            status["notes"].append(file.name)
    
    # Check audio
    for file in dirs["audio_dir"].glob("*"):
        if file.is_file():
            status["audio"].append(file.name)
    
    # Check processed
    for file in dirs["processed_dir"].glob("*"):
        if file.is_file():
            status["processed"].append(file.name)
    
    # Check output
    for file in dirs["output_dir"].glob("*"):
        if file.is_file():
            status["output"].append(file.name)
    
    return status

def main():
    parser = argparse.ArgumentParser(description="Organize lessons and files for AI processing")
    parser.add_argument("action", choices=["create", "organize", "list", "status"], 
                       help="Action to perform")
    parser.add_argument("--lesson-name", help="Name of the lesson")
    parser.add_argument("--base-dir", default="downloads", help="Base directory")
    
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    
    if args.action == "create":
        if not args.lesson_name:
            print("ERROR: --lesson-name required for create action", file=sys.stderr)
            sys.exit(1)
        
        print(f"Creating lesson structure for: {args.lesson_name}")
        dirs = create_lesson_structure(args.lesson_name, base_dir)
        print(f"✓ Lesson structure created at: {dirs['lesson_dir']}")
        print(f"  - Presentations: {dirs['presentations_dir']}")
        print(f"  - Notes: {dirs['notes_dir']}")
        print(f"  - Audio: {dirs['audio_dir']}")
        print(f"  - Processed: {dirs['processed_dir']}")
        print(f"  - Output: {dirs['output_dir']}")
    
    elif args.action == "organize":
        if not args.lesson_name:
            print("ERROR: --lesson-name required for organize action", file=sys.stderr)
            sys.exit(1)
        
        print(f"Organizing files for lesson: {args.lesson_name}")
        moved_files = organize_existing_files(args.lesson_name, base_dir)
        
        print(f"\n✓ Organization complete!")
        print(f"  - Presentations moved: {len(moved_files['presentations'])}")
        print(f"  - Notes moved: {len(moved_files['notes'])}")
        print(f"  - Audio moved: {len(moved_files['audio'])}")
    
    elif args.action == "list":
        lessons = list_lessons(base_dir)
        if lessons:
            print("Existing lessons:")
            for lesson in lessons:
                print(f"  - {lesson}")
        else:
            print("No lessons found. Create one with: python organize_lessons.py create --lesson-name \"Your Lesson Name\"")
    
    elif args.action == "status":
        if not args.lesson_name:
            print("ERROR: --lesson-name required for status action", file=sys.stderr)
            sys.exit(1)
        
        status = show_lesson_status(args.lesson_name, base_dir)
        print(f"Lesson Status: {status['lesson_name']}")
        print(f"  Presentations: {len(status['presentations'])} files")
        print(f"  Notes: {len(status['notes'])} files")
        print(f"  Audio: {len(status['audio'])} files")
        print(f"  Processed: {len(status['processed'])} files")
        print(f"  Output: {len(status['output'])} files")

if __name__ == "__main__":
    main()



