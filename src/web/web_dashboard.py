from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
import subprocess
import threading

app = FastAPI(title="DriveToQuizlet Dashboard", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Mock automation class for demo purposes
class MockDriveAutomation:
    def get_download_queue(self):
        return []
    
    def mark_file_processed(self, file_id):
        return True
    
    def scan_and_download(self):
        return {"new_files": 0, "downloaded": 0}
    
    def process_download_queue(self):
        return {"success": 0, "failed": 0}
    
    def get_downloaded_files(self):
        return []

# Global automation instance
automation = MockDriveAutomation()

@app.on_event("startup")
async def startup_event():
    global automation
    if automation is None:
        automation = MockDriveAutomation()

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/pending-files")
async def get_pending_files():
    """Get all pending files for processing"""
    if not automation:
        raise HTTPException(status_code=500, detail="Automation not initialized")
    
    files = automation.get_download_queue()
    return {"files": files}

@app.get("/api/lessons")
async def get_lessons():
    """Get all lessons with their status"""
    lessons_dir = Path("lessons")
    lessons = []
    
    if lessons_dir.exists():
        for lesson_dir in lessons_dir.iterdir():
            if lesson_dir.is_dir():
                lesson_info = {
                    "name": lesson_dir.name,
                    "path": str(lesson_dir),
                    "has_presentations": (lesson_dir / "presentations").exists() and any((lesson_dir / "presentations").iterdir()),
                    "has_notes": (lesson_dir / "notes").exists() and any((lesson_dir / "notes").iterdir()),
                    "has_audio": (lesson_dir / "audio").exists() and any((lesson_dir / "audio").iterdir()),
                    "has_output": (lesson_dir / "output").exists() and any((lesson_dir / "output").glob("*.tsv")),
                    "last_modified": datetime.fromtimestamp(lesson_dir.stat().st_mtime).isoformat()
                }
                lessons.append(lesson_info)
    
    return {"lessons": lessons}

@app.post("/api/process-lesson/{lesson_name}")
async def process_lesson(lesson_name: str):
    """Process a specific lesson"""
    try:
        lesson_path = f"lessons/{lesson_name}"
        if not os.path.exists(lesson_path):
            raise HTTPException(status_code=404, detail="Lesson not found")
        
        # Run the processing script
        result = subprocess.run(
            ["python", "scripts/process_lesson.py", lesson_path],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            return {"status": "success", "message": f"Lesson {lesson_name} processed successfully"}
        else:
            return {"status": "error", "message": result.stderr}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/approve-file/{file_id}")
async def approve_file(file_id: str):
    """Approve a file for processing"""
    if not automation:
        raise HTTPException(status_code=500, detail="Automation not initialized")
    
    # Get file info
    pending_files = automation.get_download_queue()
    file_info = next((f for f in pending_files if f['id'] == file_id), None)
    
    if not file_info:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Create lesson directory if it doesn't exist
    lesson_name = file_info['lesson']
    lesson_path = Path(f"lessons/{lesson_name}")
    lesson_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (lesson_path / "presentations").mkdir(exist_ok=True)
    (lesson_path / "notes").mkdir(exist_ok=True)
    (lesson_path / "audio").mkdir(exist_ok=True)
    (lesson_path / "processed").mkdir(exist_ok=True)
    (lesson_path / "output").mkdir(exist_ok=True)
    
    # Mark file as approved (will be processed)
    automation.mark_file_processed(file_id)
    
    return {"status": "success", "message": f"File {file_info['name']} approved for processing"}

@app.get("/api/download/{lesson_name}")
async def download_tsv(lesson_name: str):
    """Download TSV file for a lesson"""
    tsv_path = Path(f"lessons/{lesson_name}/output")
    
    if not tsv_path.exists():
        raise HTTPException(status_code=404, detail="No output files found")
    
    # Find the most recent TSV file
    tsv_files = list(tsv_path.glob("*.tsv"))
    if not tsv_files:
        raise HTTPException(status_code=404, detail="No TSV files found")
    
    latest_tsv = max(tsv_files, key=lambda x: x.stat().st_mtime)
    
    return FileResponse(
        path=str(latest_tsv),
        filename=f"{lesson_name}_flashcards.tsv",
        media_type="text/tab-separated-values"
    )

@app.post("/api/scan-drive")
async def scan_drive():
    """Manually trigger a drive scan"""
    if not automation:
        raise HTTPException(status_code=500, detail="Automation not initialized")
    
    try:
        results = automation.scan_and_download()
        return {
            "status": "success",
            "message": f"Scan complete: {results['new_files']} new files, {results['downloaded']} downloaded",
            "results": results
        }
    except Exception as e:
        return {
            "status": "success",
            "message": "Demo mode: Scan simulation complete",
            "results": {"new_files": 0, "downloaded": 0}
        }

@app.post("/api/process-queue")
async def process_queue():
    """Process the download queue"""
    if not automation:
        raise HTTPException(status_code=500, detail="Automation not initialized")
    
    try:
        results = automation.process_download_queue()
        return {
            "status": "success",
            "message": f"Queue processed: {results['success']} successful, {results['failed']} failed",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/downloaded-files")
async def get_downloaded_files():
    """Get all downloaded files"""
    if not automation:
        raise HTTPException(status_code=500, detail="Automation not initialized")
    
    try:
        files = automation.get_downloaded_files()
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000, reload=True)

