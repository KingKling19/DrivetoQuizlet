#!/usr/bin/env python3
"""
Batch Lesson Processing

Process multiple lessons automatically, generating flashcards for each.
"""

import argparse
import sys
from pathlib import Path
from typing import List

def find_lessons(base_dir: Path) -> List[Path]:
    """Find all lesson directories."""
    lessons_dir = base_dir / "lessons"
    if not lessons_dir.exists():
        return []
    
    lessons = []
    for lesson_dir in lessons_dir.iterdir():
        if lesson_dir.is_dir():
            lessons.append(lesson_dir)
    
    return sorted(lessons)

def main():
    parser = argparse.ArgumentParser(description="Batch process multiple lessons")
    parser.add_argument("--base-dir", default="downloads", help="Base directory")
    parser.add_argument("--lesson", help="Process specific lesson only")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    
    if args.lesson:
        # Process specific lesson
        lesson_dir = base_dir / "lessons" / args.lesson
        if not lesson_dir.exists():
            print(f"ERROR: Lesson not found: {lesson_dir}", file=sys.stderr)
            sys.exit(1)
        
        lessons = [lesson_dir]
    else:
        # Find all lessons
        lessons = find_lessons(base_dir)
        if not lessons:
            print("No lessons found. Create lessons first with:")
            print("  python organize_lessons.py create --lesson-name \"Your Lesson Name\"")
            sys.exit(1)
    
    print(f"Found {len(lessons)} lesson(s) to process:")
    for lesson in lessons:
        print(f"  - {lesson.name}")
    
    if args.dry_run:
        print("\nDRY RUN - No processing will be performed")
        return
    
    # Process each lesson
    for i, lesson_dir in enumerate(lessons, 1):
        print(f"\n{'='*60}")
        print(f"PROCESSING LESSON {i}/{len(lessons)}: {lesson_dir.name}")
        print(f"{'='*60}")
        
        try:
            from src.processing.process_lesson import LessonProcessor
            processor = LessonProcessor()
            success = processor.process_lesson(lesson_dir)
            
            if success:
                print(f"✓ Lesson {lesson_dir.name} processed successfully")
            else:
                print(f"✗ Lesson {lesson_dir.name} processing failed")
                
        except Exception as e:
            print(f"✗ Error processing lesson {lesson_dir.name}: {e}")
    
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()




