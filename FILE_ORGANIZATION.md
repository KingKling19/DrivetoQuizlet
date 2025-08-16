# DriveToQuizlet - File Organization Guide

## Overview
This document describes the organized file structure of the DriveToQuizlet project. All files have been reorganized by purpose to make the codebase easier to navigate and understand.

## Directory Structure

```
DriveToQuizlet/
├── src/                           # Source code organized by purpose
│   ├── processing/                # Data processing and lesson pipeline
│   ├── web/                       # Web interface and dashboard
│   ├── data/                      # Database and data management
│   ├── analysis/                  # Analysis and optimization
│   ├── drive/                     # Google Drive integration
│   └── utils/                     # Utility scripts and tools
├── tests/                         # All test files
├── examples/                      # Demo scripts and examples
├── config/                        # Configuration files
├── docs/                          # Documentation
├── templates/                     # HTML templates
├── static/                        # CSS, JS, and static assets
├── lessons/                       # Lesson data and content
├── outputs/                       # Generated outputs
└── [core files]                   # Main application entry points
```

## Core Application Files (Root Level)

- **`start_dashboard.py`** - Main application launcher
- **`check_db.py`** - Database verification utility
- **`README.md`** - Project overview and setup instructions

## Source Code Organization (`src/`)

### Processing Pipeline (`src/processing/`)
Files responsible for processing lessons, audio, notes, and generating flashcards:

- **`process_lesson.py`** - Main lesson processing workflow
- **`process_lesson_optimized.py`** - Optimized lesson processing
- **`batch_process_lessons.py`** - Batch processing multiple lessons
- **`audio_processor.py`** - Audio transcription and processing
- **`notes_processor.py`** - Handwritten notes OCR processing
- **`convert_folder_to_quizlet.py`** - Generate flashcards from lesson content
- **`flashcard_optimizer.py`** - Flashcard quality optimization
- **`batch_flashcard_optimizer.py`** - Batch flashcard optimization
- **`context_optimizer.py`** - Content context optimization
- **`integrate_powerpoint_audio.py`** - PowerPoint audio integration
- **`integrate_powerpoint_notes.py`** - PowerPoint notes integration

### Web Interface (`src/web/`)
Web dashboard and user interface components:

- **`enhanced_dashboard.py`** - Main web application (FastAPI)
- **`web_dashboard.py`** - Alternative dashboard implementation
- **`launch_dashboard.py`** - Dashboard launcher utility
- **`flashcard_review_interface.py`** - Flashcard review and management UI

### Data Management (`src/data/`)
Database operations, content indexing, and model management:

- **`quizlet_database.py`** - Database operations and schema
- **`lesson_content_indexer.py`** - Content indexing and search
- **`cross_lesson_analyzer.py`** - Cross-lesson relationship analysis
- **`model_manager.py`** - AI model management and loading

### Analysis & Optimization (`src/analysis/`)
Quality assessment, clustering, and performance monitoring:

- **`flashcard_clustering.py`** - Flashcard clustering and grouping
- **`flashcard_quality_assessor.py`** - Quality assessment algorithms
- **`lesson_relationship_visualizer.py`** - Lesson relationship visualization
- **`performance_monitor.py`** - Performance tracking and metrics
- **`optimized_file_operations.py`** - Optimized file operations

### Google Drive Integration (`src/drive/`)
Google Drive automation and monitoring:

- **`drive_automation.py`** - Main Google Drive automation
- **`drive_cli.py`** - Command-line interface for Drive operations
- **`drive_monitor.py`** - Monitor Drive for changes
- **`drive_list_folder.py`** - List and browse Drive folders
- **`drive_test.py`** - Drive integration testing
- **`desktop_monitor.py`** - Desktop monitoring for Drive sync

### Utilities (`src/utils/`)
Helper scripts and maintenance tools:

- **`organize_lessons.py`** - Lesson organization utilities
- **`create_tlp_flashcards.py`** - TLP-specific flashcard creation
- **`update_tlp_lesson_names.py`** - TLP lesson name updates

## Testing (`tests/`)
All test files have been moved to the `tests/` directory:

- **`test_phase1_implementation.py`** - Phase 1 testing
- **`test_phase2_implementation.py`** - Phase 2 testing
- **`test_phase3_implementation.py`** - Phase 3 testing
- **`test_flashcard_review_interface.py`** - UI testing
- **`test_performance.py`** - Performance testing
- **`test_new_lesson_creation.py`** - Lesson creation testing
- And more...

## Examples (`examples/`)
Demo scripts and usage examples:

- **`demo_phase1.py`** - Phase 1 demonstration
- **`demo_phase3_features.py`** - Phase 3 features demo

## Configuration (`config/`)
Configuration files and settings:

- **`semantic_embeddings.pkl`** - Semantic embeddings cache
- **Various JSON configs** - Application configuration files

## Documentation (`docs/`)
Project documentation organized by topic:

- **`features/`** - Feature specifications
- **`commands/`** - Command documentation
- **Implementation summaries** - Phase implementation details
- **Performance guides** - Optimization documentation

## Web Assets
- **`templates/`** - HTML templates for the web interface
- **`static/`** - CSS, JavaScript, and static assets

## Data Directories
- **`lessons/`** - Lesson content and data
- **`outputs/`** - Generated files and outputs

## Import Path Changes

Due to the reorganization, import statements have been updated throughout the codebase:

### Old Format:
```python
from scripts.flashcard_optimizer import FlashcardOptimizer
```

### New Format:
```python
from src.processing.flashcard_optimizer import FlashcardOptimizer
```

### Key Import Mappings:
- `scripts.web_dashboard` → `src.web.web_dashboard`
- `scripts.enhanced_dashboard` → `src.web.enhanced_dashboard`
- `scripts.flashcard_optimizer` → `src.processing.flashcard_optimizer`
- `scripts.model_manager` → `src.data.model_manager`
- `scripts.performance_monitor` → `src.analysis.performance_monitor`
- `scripts.drive_automation` → `src.drive.drive_automation`

## Benefits of This Organization

1. **Clear Purpose Separation** - Each directory has a specific purpose
2. **Easier Navigation** - Files are grouped logically
3. **Better Maintainability** - Related files are co-located
4. **Scalability** - Easy to add new files in appropriate locations
5. **Professional Structure** - Follows software engineering best practices

## Running the Application

The main entry point remains the same:
```bash
python start_dashboard.py
```

All imports have been updated to work with the new structure automatically.

## Finding Files

Use this guide to quickly locate files by their purpose:

- **Need to modify processing logic?** → `src/processing/`
- **Working on the web interface?** → `src/web/`
- **Database or data operations?** → `src/data/`
- **Performance or quality analysis?** → `src/analysis/`
- **Google Drive features?** → `src/drive/`
- **Adding tests?** → `tests/`
- **Creating examples?** → `examples/`
- **Configuration changes?** → `config/`

This organization makes the DriveToQuizlet codebase much more maintainable and easier to understand for both current and future development work.