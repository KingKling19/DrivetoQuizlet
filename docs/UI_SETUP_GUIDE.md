# DriveToQuizlet - User Interface Setup Guide

## 🎨 Enhanced Web Dashboard

The DriveToQuizlet application now features a modern, responsive web dashboard with enhanced functionality and user experience improvements.

## 🚀 Quick Start

### Method 1: Using the Startup Script (Recommended)
```bash
cd /workspace
python3 start_dashboard.py
```

### Method 2: Manual Startup
```bash
cd /workspace
python3 -m uvicorn scripts.web_dashboard:app --host localhost --port 8000 --reload
```

The dashboard will be available at: **http://localhost:8000**

## ✨ New Features

### 🎯 Enhanced User Experience
- **Modern animations** with slide-in effects and smooth transitions
- **Real-time search** functionality for lessons
- **Keyboard shortcuts** for quick navigation:
  - `Ctrl/Cmd + R`: Refresh data
  - `Ctrl/Cmd + S`: Scan Google Drive
  - `1`, `2`, `3`: Switch between tabs
- **Auto-refresh** with configurable settings
- **Enhanced notifications** with different types (success, error, info, warning)
- **Loading states** with progress indicators

### 📱 Responsive Design
- **Mobile-friendly** layout that adapts to different screen sizes
- **Touch-friendly** buttons and interactions
- **Accessibility improvements** with focus indicators and screen reader support

### 🎨 Visual Enhancements
- **File type-specific colors** for better visual categorization
- **Status indicators** with animated icons
- **Progress cards** with shimmer effects
- **Custom scrollbars** for a polished look
- **Dark mode support** (auto-detection)

### ⚙️ Advanced Functionality
- **Settings panel** with auto-refresh toggle
- **Search functionality** to filter lessons
- **Animated counters** for statistics
- **Better error handling** with user-friendly messages
- **Browser history integration** for tab navigation

## 📁 File Structure

```
static/
├── css/
│   └── dashboard.css        # Enhanced styles and animations
├── js/
│   └── dashboard.js         # Modern JavaScript functionality
└── images/                  # Placeholder for future assets

templates/
└── dashboard.html           # Main dashboard template

scripts/
└── web_dashboard.py         # FastAPI backend server
```

## 🛠️ Development

### Dependencies
The UI requires these Python packages:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `jinja2` - Template engine
- `python-multipart` - File upload support

Install with:
```bash
pip install -r requirements.txt
```

### Development Mode
For development with auto-reload:
```bash
python3 start_dashboard.py
```

### Production Mode
For production deployment:
```bash
python3 -m uvicorn scripts.web_dashboard:app --host 0.0.0.0 --port 8000
```

## 🎯 User Interface Features

### 📊 Dashboard Overview
- **Statistics cards** showing pending files, active lessons, and ready downloads
- **Real-time updates** every 30 seconds (configurable)
- **Last scan timestamp** with relative time formatting

### 📁 File Management
- **Pending files tab** with approval workflow
- **File type indicators** (presentations, notes, audio)
- **Batch approval** functionality
- **Animated file removal** when approved

### 🎓 Lesson Management
- **Lesson cards** with completion status
- **Feature indicators** showing available content types
- **Processing buttons** for incomplete lessons
- **Download buttons** for completed lessons
- **Search functionality** to find specific lessons

### 🔧 Settings & Configuration
- **Auto-refresh toggle** for real-time updates
- **Theme selection** (light, dark, auto)
- **Keyboard shortcuts** reference
- **Notification preferences**

## 🎨 Customization

### Colors & Themes
The dashboard uses a color system based on file types:
- **Blue**: Presentations and processing states
- **Green**: Notes and completion states  
- **Purple**: Audio files and download actions
- **Yellow**: Pending states
- **Red**: Error states

### Animation Preferences
Users who prefer reduced motion can disable animations through their browser's accessibility settings, and the dashboard will respect these preferences.

## 🔧 Troubleshooting

### Common Issues

1. **Server won't start**
   - Check that all dependencies are installed
   - Ensure port 8000 is not in use
   - Verify you're in the correct directory

2. **Static files not loading**
   - Confirm the `static/` directory exists
   - Check file permissions
   - Restart the server

3. **API errors**
   - Verify Google Drive credentials are configured
   - Check database connectivity
   - Review server logs for details

### Browser Compatibility
The dashboard is tested and optimized for:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 📚 API Reference

### Endpoints
- `GET /` - Dashboard home page
- `GET /api/pending-files` - Get files awaiting approval
- `GET /api/lessons` - Get all lessons with status
- `POST /api/scan-drive` - Manually trigger drive scan
- `POST /api/approve-file/{file_id}` - Approve a specific file
- `POST /api/process-lesson/{lesson_name}` - Process a lesson
- `GET /api/download/{lesson_name}` - Download lesson TSV

### Response Format
All API endpoints return JSON responses with this structure:
```json
{
    "status": "success|error",
    "message": "Human readable message",
    "data": { /* endpoint-specific data */ }
}
```

## 🚀 Future Enhancements

Planned improvements include:
- **Drag & drop** file upload
- **Progress tracking** for long-running processes
- **Lesson preview** functionality
- **Export options** (JSON, CSV, PDF)
- **User management** and authentication
- **Advanced search** with filters
- **Bulk operations** interface

## 📞 Support

For issues or questions about the UI:
1. Check this documentation
2. Review the browser console for errors
3. Check server logs for backend issues
4. Refer to the main README.md for general application support