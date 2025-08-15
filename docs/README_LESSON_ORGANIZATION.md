# 🎓 Lesson Organization System

This system organizes your military training materials by lesson and automatically processes them to create high-quality Quizlet flashcards.

## 📁 File Organization Structure

```
downloads/
├── lessons/
│   ├── Lesson_Name_1/
│   │   ├── presentations/     # PowerPoint files (.pptx)
│   │   ├── notes/            # Handwritten notes (.png, .jpg)
│   │   ├── audio/            # Lecture recordings (.m4a, .mp3)
│   │   ├── processed/        # AI processing results
│   │   ├── output/           # Final Quizlet flashcards (.tsv)
│   │   └── README.md         # Lesson-specific instructions
│   └── Lesson_Name_2/
│       └── ... (same structure)
```

## 🚀 Quick Start Guide

### 1. Create a New Lesson

```bash
# Create lesson structure
python organize_lessons.py create --lesson-name "Perform Effectively In An Operational Environment"
```

### 2. Add Your Files

Place your files in the appropriate folders:
- **PowerPoint files** → `lessons/Lesson_Name/presentations/`
- **Handwritten notes** → `lessons/Lesson_Name/notes/`
- **Audio recordings** → `lessons/Lesson_Name/audio/`

### 3. Process the Lesson

```bash
# Process single lesson
python process_lesson.py "downloads/lessons/Lesson_Name"

# Or process all lessons
python batch_process_lessons.py
```

## 📋 Available Commands

### Lesson Organization

```bash
# Create new lesson structure
python organize_lessons.py create --lesson-name "Your Lesson Name"

# Organize existing files into lesson structure
python organize_lessons.py organize --lesson-name "Your Lesson Name"

# List all lessons
python organize_lessons.py list

# Check lesson status
python organize_lessons.py status --lesson-name "Your Lesson Name"
```

### Lesson Processing

```bash
# Process single lesson
python process_lesson.py "downloads/lessons/Lesson_Name"

# Batch process all lessons
python batch_process_lessons.py

# Process specific lesson in batch
python batch_process_lessons.py --lesson "Lesson_Name"

# Dry run (see what would be processed)
python batch_process_lessons.py --dry-run
```

## 🔄 Processing Workflow

The system automatically:

1. **Scans** lesson directory for available files
2. **Processes** PowerPoint files → Flashcards
3. **Processes** handwritten notes → OCR text + AI analysis
4. **Processes** audio files → Transcription + AI summary
5. **Integrates** all sources → Comprehensive flashcards
6. **Outputs** Quizlet-ready TSV files

## 📊 Output Files

Each lesson generates:

- `output/Lesson_Name_flashcards.tsv` - **Ready for Quizlet import**
- `output/Lesson_Name_flashcards.json` - Complete data with metadata
- `processed/` - All intermediate processing results

## 🎯 Example Workflow

### Step 1: Create Lesson Structure
```bash
python organize_lessons.py create --lesson-name "Troop Leading Procedures"
```

### Step 2: Add Files
- Copy `TLP.pptx` to `downloads/lessons/Troop_Leading_Procedures/presentations/`
- Copy note images to `downloads/lessons/Troop_Leading_Procedures/notes/`
- Copy audio files to `downloads/lessons/Troop_Leading_Procedures/audio/`

### Step 3: Process
```bash
python process_lesson.py "downloads/lessons/Troop_Leading_Procedures"
```

### Step 4: Import to Quizlet
- Open Quizlet
- Create new set
- Import from `output/Troop_Leading_Procedures_flashcards.tsv`

## 🔧 Advanced Usage

### Organize Existing Files
If you already have files scattered in different folders:

```bash
# This will find and organize files by lesson name
python organize_lessons.py organize --lesson-name "Your Lesson Name"
```

### Batch Processing
Process multiple lessons at once:

```bash
# Process all lessons
python batch_process_lessons.py

# See what would be processed
python batch_process_lessons.py --dry-run
```

### Check Lesson Status
See what files are available for a lesson:

```bash
python organize_lessons.py status --lesson-name "Your Lesson Name"
```

## 📝 File Naming Conventions

- **Lesson names**: Use descriptive names like "Troop Leading Procedures" or "Operational Environment"
- **PowerPoint files**: Include lesson name in filename for automatic detection
- **Note images**: Include lesson name in filename for automatic detection
- **Audio files**: Include lesson name in filename for automatic detection

## 🎯 Best Practices

1. **Use descriptive lesson names** that match your course structure
2. **Include lesson name in filenames** for automatic organization
3. **Process lessons as you complete them** rather than all at once
4. **Review generated flashcards** before importing to Quizlet
5. **Keep original files** in the organized structure for future reference

## 🔍 Troubleshooting

### No files found
- Check that files are in the correct folders
- Ensure filenames contain the lesson name
- Use `python organize_lessons.py status --lesson-name "Lesson"` to check

### Processing errors
- Ensure all required Python packages are installed
- Check that your `.env` file contains `OPENAI_API_KEY`
- Verify file permissions and disk space

### Import issues
- Ensure TSV files are properly formatted
- Check Quizlet import requirements
- Verify no special characters in terms/definitions

## 📈 Scaling Up

As you add more lessons:

1. **Create lesson structure** for each new lesson
2. **Add files** to appropriate folders
3. **Process individually** or use batch processing
4. **Organize output** by importing to Quizlet sets

This system scales to handle hundreds of lessons while maintaining organization and quality.




