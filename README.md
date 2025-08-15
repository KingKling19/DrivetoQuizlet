# Army ADA BOLC Flashcard App

An intelligent flashcard generation system designed for Army Air Defense Artillery Basic Officer Leader Course (ADA BOLC) training materials. This application automatically processes PowerPoint presentations, handwritten notes, and audio recordings to create high-quality, context-aware flashcards.

## 🎯 Features

### ✅ Completed Features
- **Google Drive Integration**: Automatic file monitoring and download from Google Drive
- **Multi-format Content Processing**: PowerPoint (.pptx), handwritten notes (PNG/JPG), and audio files (.m4a, .mp3)
- **AI-Powered Content Analysis**: Uses OpenAI GPT-4o-mini for intelligent content extraction
- **Cross-Lesson Context System**: Advanced similarity analysis and relationship mapping between lessons
- **Web Dashboard**: FastAPI-based interface for monitoring and management
- **Military Context Awareness**: Specialized processing for military terminology and concepts
- **Performance Monitoring**: Real-time system performance tracking

### 🔄 In Development
- **Flashcard Optimization & Refinement**: Advanced duplicate detection and quality assessment
- **Quizlet API Integration**: Direct upload to Quizlet platform
- **Automated Workflow System**: End-to-end processing automation

## 🚀 Quick Start

### 1. Initial Setup

```bash
# Clone the repository (if applicable)
# cd DriveToQuizlet

# Run the setup script
python3 setup.py
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 3. Configuration

1. **Environment Variables**: Copy `.env.template` to `.env` and fill in your values:
   ```bash
   cp .env.template .env
   # Edit .env with your API keys and configuration
   ```

2. **OpenAI API**: Add your OpenAI API key to `.env`:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Google Drive API**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing
   - Enable Google Drive API
   - Create OAuth 2.0 credentials
   - Download `credentials.json` to the `config/` directory

### 4. Authentication Test

```bash
# Test Google Drive authentication
python scripts/drive_test.py
```

### 5. Configure Drive Folders

Edit `config/drive_config.json` with your Google Drive folder IDs:
```json
{
  "drive_folders": {
    "personal_notes_audio": "your_folder_id_here",
    "instructor_presentations": "your_folder_id_here"
  }
}
```

## 📖 Usage

### Processing a Single Lesson

```bash
# Process PowerPoint files in a directory
python scripts/convert_folder_to_quizlet.py "/path/to/lesson/presentations"

# Process with specific model and settings
python scripts/convert_folder_to_quizlet.py "/path/to/lesson/presentations" \
  --model gpt-4o-mini \
  --window 3 \
  --verbose
```

### Cross-Lesson Context Analysis

```bash
# Index all lessons for context analysis
python scripts/lesson_content_indexer.py

# Analyze relationships between lessons
python scripts/cross_lesson_analyzer.py --analyze

# Optimize context for specific lesson
python scripts/context_optimizer.py --optimize "TLP"
```

### Web Dashboard

```bash
# Start the web dashboard
python scripts/enhanced_dashboard.py

# Or use the launcher
python scripts/launch_dashboard.py
```

Visit `http://localhost:8000` to access the dashboard.

### Automated Processing

```bash
# Monitor Google Drive and process new files
python scripts/drive_monitor.py

# Batch process multiple lessons
python scripts/batch_process_lessons.py
```

## 🏗️ Project Structure

```
DriveToQuizlet/
├── scripts/                    # Main processing scripts
│   ├── convert_folder_to_quizlet.py    # Core flashcard generation
│   ├── lesson_content_indexer.py       # Content indexing for cross-lesson context
│   ├── cross_lesson_analyzer.py        # Relationship analysis
│   ├── context_optimizer.py            # Context optimization
│   ├── drive_automation.py             # Google Drive integration
│   ├── enhanced_dashboard.py           # Web dashboard
│   └── ...
├── config/                     # Configuration files
│   ├── drive_config.json      # Google Drive settings
│   ├── credentials.json       # Google API credentials
│   └── ...
├── lessons/                    # Lesson content (organized by topic)
│   ├── TLP/
│   ├── Conducting_Operations_in_a_Degraded_Space/
│   └── Perform_Effectively_In_An_Operational_Environment/
├── outputs/                    # Generated flashcards and reports
├── logs/                      # Application logs
├── static/                    # Web dashboard assets
├── templates/                 # HTML templates
├── requirements.txt           # Python dependencies
├── .env.template             # Environment variables template
└── setup.py                  # Setup script
```

## 🔧 Configuration Options

### Processing Parameters

- `DEFAULT_MODEL`: OpenAI model to use (default: gpt-4o-mini)
- `MAX_RETRIES`: API retry attempts (default: 4)
- `MIN_DEF_LEN`: Minimum definition length (default: 12)
- `WINDOW_SIZE`: Slides per LLM window (default: 3)

### Cross-Lesson Context

- `semantic_weight`: Weight for semantic similarity (default: 0.4)
- `concept_overlap_weight`: Weight for concept overlap (default: 0.3)
- `relationship_strength_weight`: Weight for relationship strength (default: 0.2)
- `content_freshness_weight`: Weight for content freshness (default: 0.1)

## 📊 Performance Monitoring

The system includes comprehensive performance monitoring:

```bash
# View performance metrics
python scripts/performance_monitor.py

# Run performance tests
python scripts/test_performance.py
```

## 🧪 Testing

```bash
# Test individual components
python scripts/test_phase1_implementation.py
python scripts/test_phase2_implementation.py
python scripts/test_phase3_implementation.py

# Test new lesson creation
python scripts/test_new_lesson_creation.py
```

## 🛠️ Development

### Adding New Content Processors

1. Create processor in `scripts/`
2. Follow the pattern in existing processors
3. Add error handling and logging
4. Update configuration as needed

### Extending Cross-Lesson Context

1. Modify `lesson_content_indexer.py` for new content types
2. Update `cross_lesson_analyzer.py` for new relationship types
3. Enhance `context_optimizer.py` for new optimization factors

## 📝 Command Reference

### Core Processing
- `convert_folder_to_quizlet.py`: Generate flashcards from PowerPoint files
- `audio_processor.py`: Process audio recordings with Whisper
- `notes_processor.py`: Extract text from handwritten notes with OCR
- `integrate_powerpoint_notes.py`: Combine PowerPoint and notes
- `integrate_powerpoint_audio.py`: Combine PowerPoint and audio

### Cross-Lesson Context
- `lesson_content_indexer.py`: Index lesson content for analysis
- `cross_lesson_analyzer.py`: Analyze relationships between lessons
- `context_optimizer.py`: Optimize context selection

### System Management
- `drive_automation.py`: Google Drive integration
- `enhanced_dashboard.py`: Web dashboard
- `performance_monitor.py`: System monitoring
- `batch_process_lessons.py`: Automated processing

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Errors**: Check API key and quota limits
2. **Google Drive Access**: Verify credentials and folder permissions
3. **OCR Issues**: Ensure easyocr is properly installed
4. **Audio Processing**: Check Whisper model installation

### Debug Mode

Add `--verbose` flag to most scripts for detailed logging:
```bash
python scripts/convert_folder_to_quizlet.py "/path/to/lesson" --verbose
```

### Logs

Check application logs in the `logs/` directory for detailed error information.

## 📄 License

This project is designed for educational use in military training environments.

## 🤝 Contributing

1. Follow existing code patterns and documentation
2. Add comprehensive error handling
3. Include logging for debugging
4. Test with sample data before deployment
5. Update documentation for new features

## 📞 Support

For issues and questions:
1. Check the logs in `logs/` directory
2. Run diagnostic scripts in `scripts/test_*.py`
3. Review configuration files in `config/`
4. Consult the performance monitor for system status

---

**Version**: 1.0  
**Last Updated**: December 2024  
**Compatibility**: Python 3.8+

