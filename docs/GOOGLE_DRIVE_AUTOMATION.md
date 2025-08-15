# 🚀 Google Drive Automation Setup Guide

This guide will help you set up automatic file downloading and organization from Google Drive to your DriveToQuizlet system.

## 📋 Prerequisites

1. **Google Drive API Access**: You need access to Google Drive API
2. **Python Environment**: Your DriveToQuizlet environment should be set up
3. **Google Cloud Project**: A Google Cloud project with Drive API enabled

## 🔧 Step 1: Google Cloud Setup

### 1.1 Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Google Drive API:
   - Go to "APIs & Services" > "Library"
   - Search for "Google Drive API"
   - Click "Enable"

### 1.2 Create Credentials
1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth 2.0 Client IDs"
3. Choose "Desktop application"
4. Download the credentials file as `credentials.json`
5. Place it in your `config/` folder

## 🔐 Step 2: Authentication Setup

### 2.1 First-Time Authentication
```bash
# Navigate to your DriveToQuizlet directory
cd /path/to/DriveToQuizlet

# Run the authentication script
python scripts/drive_test.py
```

This will:
- Open a browser window for Google authentication
- Ask you to log in to your Google account
- Grant permissions to access your Drive
- Create `config/token.json` with your credentials

### 2.2 Verify Authentication
```bash
# Test the connection
python scripts/drive_cli.py test
```

You should see: `✅ Google Drive connection successful!`

## ⚙️ Step 3: Configure Drive Folders

### 3.1 Get Folder IDs
1. Open Google Drive in your browser
2. Navigate to the folder you want to monitor
3. Copy the folder ID from the URL:
   ```
   https://drive.google.com/drive/folders/FOLDER_ID_HERE
   ```

### 3.2 Update Configuration
Edit `config/drive_config.json`:

```json
{
  "drive_folders": {
    "personal_notes_audio": "YOUR_PERSONAL_FOLDER_ID",
    "instructor_presentations": "YOUR_INSTRUCTOR_FOLDER_ID"
  },
  "file_types": {
    "audio": ["audio/x-m4a", "audio/mp4", "audio/mpeg"],
    "presentations": [
      "application/vnd.openxmlformats-officedocument.presentationml.presentation",
      "application/vnd.ms-powerpoint",
      "application/vnd.google-apps.presentation"
    ],
    "notes": ["image/jpeg", "image/jpg", "image/png"]
  },
  "processing": {
    "check_interval_hours": 1,
    "auto_download": true,
    "auto_process": false,
    "require_approval": true
  }
}
```

## 🚀 Step 4: Test the System

### 4.1 Test Configuration
```bash
# Show current configuration
python scripts/drive_cli.py config
```

### 4.2 Test Single Scan
```bash
# Perform a single scan
python scripts/drive_cli.py scan
```

### 4.3 Check Results
```bash
# Show download queue
python scripts/drive_cli.py queue

# Show downloaded files
python scripts/drive_cli.py downloaded
```

## 🎯 Step 5: Start Automation

### 5.1 Manual Mode (Recommended for Testing)
```bash
# Start the automation service
python scripts/drive_cli.py start
```

This will:
- Scan your configured folders every hour
- Download new files automatically
- Organize them into lesson folders
- Show progress and status

### 5.2 Web Dashboard Mode
```bash
# Start the web dashboard
python scripts/web_dashboard.py
```

Then open: `http://localhost:8000`

## 📁 File Organization

The system automatically organizes files into this structure:

```
lessons/
├── Lesson_Name_1/
│   ├── presentations/     # PowerPoint files
│   ├── notes/            # Handwritten notes (images)
│   ├── audio/            # Audio recordings
│   ├── processed/        # AI processing results
│   └── output/           # Final Quizlet flashcards
└── Lesson_Name_2/
    └── ... (same structure)
```

## 🔍 How It Works

### 1. **Intelligent File Detection**
- Scans configured Google Drive folders
- Detects lesson names from filenames
- Categorizes files by type (presentations, notes, audio)

### 2. **Automatic Organization**
- Creates lesson folders automatically
- Places files in appropriate subfolders
- Maintains original filenames

### 3. **Progress Tracking**
- Shows download progress for large files
- Tracks successful and failed downloads
- Maintains database of processed files

### 4. **Error Handling**
- Retries failed downloads
- Logs errors for debugging
- Continues processing other files

## 🛠️ CLI Commands

| Command | Description |
|---------|-------------|
| `test` | Test Google Drive connection |
| `scan` | Perform single scan and download |
| `queue` | Show download queue |
| `process` | Process download queue manually |
| `downloaded` | Show downloaded files |
| `config` | Show current configuration |
| `start` | Start automation service |

## 🌐 Web Dashboard Features

### Dashboard Sections
1. **Drive Status**: Connection and last scan info
2. **Pending Files**: Files waiting to be downloaded
3. **Downloaded Files**: Successfully downloaded files
4. **Lessons**: Organized lesson folders
5. **Actions**: Manual scan and process buttons

### API Endpoints
- `POST /api/scan-drive` - Trigger manual scan
- `POST /api/process-queue` - Process download queue
- `GET /api/pending-files` - Get pending files
- `GET /api/downloaded-files` - Get downloaded files

## 🔧 Configuration Options

### Processing Settings
```json
{
  "processing": {
    "check_interval_hours": 1,        // How often to scan
    "auto_download": true,            // Auto-download new files
    "auto_process": false,            // Auto-process after download
    "require_approval": true          // Require manual approval
  }
}
```

### File Type Support
```json
{
  "file_types": {
    "audio": ["audio/x-m4a", "audio/mp4"],
    "presentations": ["application/vnd.openxmlformats-officedocument.presentationml.presentation"],
    "notes": ["image/jpeg", "image/jpg", "image/png"]
  }
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```bash
   # Re-authenticate
   python scripts/drive_test.py
   ```

2. **Folder Not Found**
   - Check folder IDs in `config/drive_config.json`
   - Ensure you have access to the folders
   - Verify folder IDs are correct

3. **No Files Detected**
   - Check file types are supported
   - Verify files are in configured folders
   - Check file permissions

4. **Download Failures**
   - Check internet connection
   - Verify disk space
   - Check file permissions

### Debug Mode
```bash
# Enable debug logging
export PYTHONPATH=.
python -u scripts/drive_cli.py scan
```

## 📊 Monitoring and Logs

### Database Files
- `config/drive_automation.db` - SQLite database with file tracking
- `config/drive_monitor.db` - Legacy monitor database

### Log Locations
- Console output for real-time status
- Database for historical tracking
- Web dashboard for visual monitoring

## 🔄 Integration with Processing

After files are downloaded:

1. **Manual Processing**:
   ```bash
   python scripts/process_lesson.py "lessons/Lesson_Name"
   ```

2. **Batch Processing**:
   ```bash
   python scripts/batch_process_lessons.py
   ```

3. **Web Dashboard**:
   - Use the "Process Lesson" button
   - Download generated TSV files

## 🎯 Best Practices

1. **Start Small**: Test with a few files first
2. **Monitor Initially**: Use manual mode until confident
3. **Regular Backups**: Backup your `config/` folder
4. **Check Logs**: Monitor for errors regularly
5. **Update Credentials**: Refresh tokens if needed

## 🚀 Advanced Features

### Custom File Types
Add new file types to `config/drive_config.json`:

```json
{
  "file_types": {
    "documents": ["application/pdf", "application/msword"],
    "spreadsheets": ["application/vnd.google-apps.spreadsheet"]
  }
}
```

### Multiple Folders
Add more folders to monitor:

```json
{
  "drive_folders": {
    "personal_notes_audio": "FOLDER_ID_1",
    "instructor_presentations": "FOLDER_ID_2",
    "additional_materials": "FOLDER_ID_3"
  }
}
```

### Scheduled Processing
Set up automatic processing after downloads:

```json
{
  "processing": {
    "auto_process": true,
    "process_delay_minutes": 5
  }
}
```

## 📞 Support

For issues:
1. Check this troubleshooting guide
2. Review console output for errors
3. Check database for file status
4. Verify Google Drive permissions

Your DriveToQuizlet system is now ready to automatically download and organize files from Google Drive! 🎉
