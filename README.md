# DriveToQuizlet - Military Training Lesson Processor

A comprehensive system for processing military training materials into Quizlet flashcards.

## 📁 Folder Structure

### `lessons/` - Your Training Materials
Each lesson has its own folder with this structure:
```
lessons/[Lesson_Name]/
├── presentations/     # PowerPoint files (.pptx)
├── notes/            # Handwritten notes (JPG/PNG)
├── audio/            # Audio recordings (.m4a, .mp3)
├── processed/        # AI-processed outputs
├── output/           # Final TSV flashcards for Quizlet
└── README.md         # Lesson documentation
```

### `scripts/` - Processing Tools
- `process_lesson.py` - Main lesson processor
- `organize_lessons.py` - Create lesson folders
- `audio_processor.py` - Process audio files
- `notes_processor.py` - Process handwritten notes
- `integrate_powerpoint_*.py` - Process PowerPoint files

### `outputs/` - Generated Flashcards
- TSV files ready for Quizlet import
- JSON exports
- Combined study materials

### `docs/` - Documentation
- Setup guides
- Performance notes
- Terminal commands

### `config/` - Configuration Files
- API keys and settings

## 🚀 Quick Start

1. **Add new lesson materials** to `lessons/[Lesson_Name]/`
2. **Run processing**: `python scripts/process_lesson.py "lessons/[Lesson_Name]"`
3. **Import to Quizlet**: Use the TSV files from `lessons/[Lesson_Name]/output/`

## 📋 Current Lessons
- Perform Effectively In An Operational Environment
- Conducting Operations in a Degraded Space  
- TLP (Troop Leading Procedures)

