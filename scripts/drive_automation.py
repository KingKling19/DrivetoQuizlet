import os
import json
import time
import sqlite3
import shutil
from datetime import datetime
from pathlib import Path
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import re
from typing import Dict, List, Optional, Tuple
import threading
from scripts.optimized_file_operations import file_ops

class DriveAutomation:
    def __init__(self):
        self.config = self.load_config()
        self.service = self.authenticate_drive()
        self.db_path = "config/drive_automation.db"
        self.init_database()
        self.lessons_dir = Path("lessons")
        self.lessons_dir.mkdir(exist_ok=True)
        
    def load_config(self):
        """Load configuration from config file"""
        config_path = Path("config/drive_config.json")
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        else:
            return {
                "drive_folders": {
                    "personal_notes_audio": "",
                    "instructor_presentations": ""
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
                    "auto_download": True,
                    "auto_process": False,
                    "require_approval": True
                }
            }
    
    def authenticate_drive(self):
        """Authenticate with Google Drive API"""
        token_path = Path("config/token.json")
        if not token_path.exists():
            raise FileNotFoundError("Google Drive token not found. Please run drive_test.py first to authenticate.")
        
        creds = Credentials.from_authorized_user_file(str(token_path), 
                                                     ['https://www.googleapis.com/auth/drive.readonly'])
        return build('drive', 'v3', credentials=creds)
    
    def init_database(self):
        """Initialize SQLite database for tracking files"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drive_files (
                file_id TEXT PRIMARY KEY,
                file_name TEXT,
                file_type TEXT,
                lesson_name TEXT,
                drive_folder TEXT,
                local_path TEXT,
                download_date TEXT,
                status TEXT,
                file_size INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS download_queue (
                file_id TEXT PRIMARY KEY,
                file_name TEXT,
                file_type TEXT,
                lesson_name TEXT,
                drive_folder TEXT,
                queued_date TEXT,
                status TEXT,
                error_message TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def detect_lesson_name(self, filename: str) -> str:
        """Intelligently detect lesson name from filename"""
        # Remove file extensions and common suffixes
        clean_name = re.sub(r'\.(pptx?|jpg|jpeg|png|m4a|mp3|pdf)$', '', filename, flags=re.IGNORECASE)
        clean_name = re.sub(r'\s*\(\d+\)\s*$', '', clean_name)  # Remove (2), (3), etc.
        
        # Normalize TLP variations to a single lesson name
        if re.search(r'TLP.*Overview.*combined', clean_name, re.IGNORECASE):
            return 'TLP'
        
        # Common military training lesson patterns
        lesson_patterns = [
            r'Perform.*Operational.*Environment',
            r'Conducting.*Operations.*Degraded.*Space',
            r'TLP.*',
            r'Troop.*Leading.*Procedures',
            r'Operational.*Environment',
            r'Degraded.*Space'
        ]
        
        for pattern in lesson_patterns:
            match = re.search(pattern, clean_name, re.IGNORECASE)
            if match:
                lesson_name = match.group(0)
                # Normalize TLP variations
                if re.search(r'TLP', lesson_name, re.IGNORECASE):
                    return 'TLP'
                return lesson_name
        
        # If no pattern matches, create a new lesson name from the filename
        # Clean up the filename to make it suitable as a folder name
        lesson_name = clean_name.strip()
        
        # Remove special characters that might cause issues with folder names
        lesson_name = re.sub(r'[<>:"/\\|?*]', '', lesson_name)
        
        # Replace multiple spaces with single space
        lesson_name = re.sub(r'\s+', ' ', lesson_name)
        
        # Limit length to avoid overly long folder names
        if len(lesson_name) > 50:
            lesson_name = lesson_name[:47] + "..."
        
        # If the lesson name is empty or just whitespace, use a default
        if not lesson_name or lesson_name.isspace():
            lesson_name = "Unknown_Lesson"
        
        print(f"  🆕 Creating new lesson folder: '{lesson_name}' for file: {filename}")
        return lesson_name
    
    def categorize_file(self, mime_type: str) -> str:
        """Categorize file based on MIME type"""
        for category, mime_types in self.config['file_types'].items():
            if mime_type in mime_types:
                return category
        return 'unknown'
    
    def get_folder_name(self, file_type: str) -> str:
        """Get the appropriate folder name for file type"""
        folder_map = {
            'audio': 'audio',
            'presentations': 'presentations',
            'notes': 'notes'
        }
        return folder_map.get(file_type, 'misc')
    
    def create_lesson_structure(self, lesson_name: str):
        """Create the standard folder structure for a new lesson"""
        lesson_dir = self.lessons_dir / lesson_name
        
        # Create main lesson directory
        lesson_dir.mkdir(exist_ok=True)
        
        # Create standard subdirectories
        subdirs = ['audio', 'presentations', 'notes', 'processed', 'output']
        for subdir in subdirs:
            (lesson_dir / subdir).mkdir(exist_ok=True)
        
        # Create a README file for the new lesson
        readme_path = lesson_dir / "README.md"
        if not readme_path.exists():
            with open(readme_path, 'w') as f:
                f.write(f"# {lesson_name}\n\n")
                f.write("## Overview\n")
                f.write("This lesson folder was automatically created by the Drive Automation system.\n\n")
                f.write("## Folder Structure\n")
                f.write("- `audio/` - Audio recordings and voice notes\n")
                f.write("- `presentations/` - PowerPoint presentations and slides\n")
                f.write("- `notes/` - Images of handwritten notes\n")
                f.write("- `processed/` - Processed and converted files\n")
                f.write("- `output/` - Generated outputs and results\n\n")
                f.write(f"## Created\n")
                f.write(f"Automatically created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"  📁 Created lesson structure for: {lesson_name}")
        return lesson_dir
    
    def scan_drive_folder(self, folder_id: str, folder_name: str) -> List[Dict]:
        """Scan a specific Google Drive folder for new files"""
        try:
            print(f"🔍 Scanning {folder_name} folder...")
            
            # Build query for all supported file types
            all_mime_types = []
            for category in self.config['file_types'].values():
                all_mime_types.extend(category)
            
            mime_query = " or ".join([f"mimeType = '{mime}'" for mime in all_mime_types])
            query = f"'{folder_id}' in parents and ({mime_query}) and trashed = false"
            
            results = self.service.files().list(
                q=query,
                fields="files(id, name, mimeType, modifiedTime, size)",
                orderBy="modifiedTime desc"
            ).execute()
            
            files = results.get('files', [])
            all_files = []
            skipped_count = 0
            
            for file in files:
                file_id = file['id']
                file_name = file['name']
                mime_type = file['mimeType']
                
                # Check if file is already downloaded
                if self.is_file_downloaded(file_id):
                    skipped_count += 1
                    continue
                
                file_type = self.categorize_file(mime_type)
                lesson_name = self.detect_lesson_name(file_name)
                
                if file_type != 'unknown':
                    all_files.append({
                        'id': file_id,
                        'name': file_name,
                        'type': file_type,
                        'lesson': lesson_name,
                        'mime_type': mime_type,
                        'drive_folder': folder_name,
                        'size': int(file.get('size', 0))
                    })
            
            print(f"  Found {len(all_files)} new files, skipped {skipped_count} already downloaded files in {folder_name}")
            return all_files
            
        except Exception as e:
            print(f"❌ Error scanning folder {folder_id}: {str(e)}")
            return []
    
    def is_file_downloaded(self, file_id: str) -> bool:
        """Check if file has already been downloaded"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT file_id, local_path, file_name FROM drive_files WHERE file_id = ?", (file_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # Check if the file actually exists locally
            local_path = result[1]
            file_name = result[2]
            
            if local_path and Path(local_path).exists():
                # Verify file size is not zero (indicating incomplete download)
                file_size = Path(local_path).stat().st_size
                if file_size > 0:
                    return True
                else:
                    print(f"  ⚠️  Found zero-byte file, will re-download: {file_name}")
                    self.remove_file_record(file_id)
                    return False
            else:
                # File was recorded but doesn't exist locally, remove from database
                print(f"  ⚠️  File recorded but missing locally, will re-download: {file_name}")
                self.remove_file_record(file_id)
                return False
        return False
    
    def remove_file_record(self, file_id: str):
        """Remove file record from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM drive_files WHERE file_id = ?", (file_id,))
        conn.commit()
        conn.close()
    
    def add_to_download_queue(self, file_info: Dict):
        """Add file to download queue"""
        # Check if file is already in queue
        if self.is_file_in_queue(file_info['id']):
            print(f"  ⏭️  Already in queue: {file_info['name']}")
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO download_queue 
            (file_id, file_name, file_type, lesson_name, drive_folder, queued_date, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            file_info['id'],
            file_info['name'],
            file_info['type'],
            file_info['lesson'],
            file_info['drive_folder'],
            datetime.now().isoformat(),
            'queued'
        ))
        
        conn.commit()
        conn.close()
    
    def is_file_in_queue(self, file_id: str) -> bool:
        """Check if file is already in download queue"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT file_id FROM download_queue WHERE file_id = ? AND status = 'queued'", (file_id,))
        result = cursor.fetchone()
        conn.close()
        return result is not None
    
    def cleanup_download_queue(self):
        """Remove files from download queue that have already been downloaded"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all queued files
        cursor.execute("SELECT file_id FROM download_queue WHERE status = 'queued'")
        queued_files = cursor.fetchall()
        
        cleaned_count = 0
        for (file_id,) in queued_files:
            if self.is_file_downloaded(file_id[0]):
                cursor.execute("DELETE FROM download_queue WHERE file_id = ?", (file_id[0],))
                cleaned_count += 1
        
        conn.commit()
        conn.close()
        
        if cleaned_count > 0:
            print(f"🧹 Cleaned up {cleaned_count} already downloaded files from queue")
    
    def download_file(self, file_info: Dict) -> Tuple[bool, str]:
        """Download a single file from Google Drive"""
        try:
            file_id = file_info['id']
            file_name = file_info['name']
            file_type = file_info['type']
            lesson_name = file_info['lesson']
            
            # Create lesson directory structure (including all subdirectories)
            lesson_dir = self.create_lesson_structure(lesson_name)
            
            # Create appropriate subfolder
            subfolder = lesson_dir / self.get_folder_name(file_type)
            subfolder.mkdir(exist_ok=True)
            
            # Determine local file path
            local_path = subfolder / file_name
            
            print(f"📥 Downloading: {file_name}")
            print(f"   📁 Destination: {local_path}")
            
            # Download file with progress tracking
            request = self.service.files().get_media(fileId=file_id)
            
            # Get file size for progress tracking
            file_size = file_info.get('size', 0)
            
            with open(local_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request, chunksize=1024*1024)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if file_size > 0:
                        progress = int(status.progress() * 100)
                        downloaded = int(status.progress() * file_size / 1024 / 1024)
                        total_mb = file_size / 1024 / 1024
                        print(f"   📊 Progress: {progress}% ({downloaded:.1f}MB / {total_mb:.1f}MB)")
            
            # Record successful download
            self.record_downloaded_file(file_info, str(local_path))
            
            print(f"✅ Downloaded successfully: {file_name}")
            return True, ""
            
        except Exception as e:
            error_msg = f"Error downloading {file_info['name']}: {str(e)}"
            print(f"❌ {error_msg}")
            self.record_download_error(file_info, str(e))
            return False, str(e)
    
    def record_downloaded_file(self, file_info: Dict, local_path: str):
        """Record successfully downloaded file"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO drive_files 
            (file_id, file_name, file_type, lesson_name, drive_folder, local_path, download_date, status, file_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            file_info['id'],
            file_info['name'],
            file_info['type'],
            file_info['lesson'],
            file_info['drive_folder'],
            local_path,
            datetime.now().isoformat(),
            'downloaded',
            file_info.get('size', 0)
        ))
        
        # Remove from download queue
        cursor.execute("DELETE FROM download_queue WHERE file_id = ?", (file_info['id'],))
        
        conn.commit()
        conn.close()
    
    def record_download_error(self, file_info: Dict, error_message: str):
        """Record download error"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE download_queue 
            SET status = 'error', error_message = ?
            WHERE file_id = ?
        ''', (error_message, file_info['id']))
        
        conn.commit()
        conn.close()
    
    def process_download_queue(self) -> Dict:
        """Process all files in the download queue"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM download_queue WHERE status = 'queued' ORDER BY queued_date")
        queued_files = cursor.fetchall()
        conn.close()
        
        if not queued_files:
            print("📭 No files in download queue")
            return {"success": 0, "failed": 0, "errors": []}
        
        print(f"🚀 Processing {len(queued_files)} files in download queue...")
        
        success_count = 0
        failed_count = 0
        errors = []
        
        for row in queued_files:
            file_info = {
                'id': row[0],
                'name': row[1],
                'type': row[2],
                'lesson': row[3],
                'drive_folder': row[4]
            }
            
            success, error = self.download_file(file_info)
            if success:
                success_count += 1
            else:
                failed_count += 1
                errors.append(f"{file_info['name']}: {error}")
        
        print(f"📊 Download Summary: {success_count} successful, {failed_count} failed")
        return {"success": success_count, "failed": failed_count, "errors": errors}
    
    def scan_and_download(self) -> Dict:
        """Scan all folders and download new files"""
        print(f"\n🔄 [{datetime.now()}] Starting Google Drive scan and download...")
        
        # Clean up download queue first
        self.cleanup_download_queue()
        
        all_new_files = []
        new_lessons_created = set()
        
        # Scan personal folder (notes and audio)
        personal_files = self.scan_drive_folder(
            self.config['drive_folders']['personal_notes_audio'],
            'personal_notes_audio'
        )
        all_new_files.extend(personal_files)
        
        # Scan instructor folder (presentations)
        instructor_files = self.scan_drive_folder(
            self.config['drive_folders']['instructor_presentations'],
            'instructor_presentations'
        )
        all_new_files.extend(instructor_files)
        
        # Track new lessons being created
        for file_info in all_new_files:
            lesson_name = file_info['lesson']
            lesson_dir = self.lessons_dir / lesson_name
            
            # Check if this is a new lesson (folder doesn't exist yet)
            if not lesson_dir.exists():
                new_lessons_created.add(lesson_name)
        
        # Add new files to download queue and show what's being queued
        for file_info in all_new_files:
            print(f"  📋 Queued: {file_info['name']} -> {file_info['lesson']} ({file_info['type']})")
            self.add_to_download_queue(file_info)
        
        # Show summary of new lessons that will be created
        if new_lessons_created:
            print(f"\n🆕 New lessons that will be created:")
            for lesson in sorted(new_lessons_created):
                print(f"   📁 {lesson}")
        
        # Process download queue if auto-download is enabled
        if self.config['processing']['auto_download']:
            download_results = self.process_download_queue()
            return {
                "new_files": len(all_new_files),
                "new_downloads": len(all_new_files),
                "new_lessons": len(new_lessons_created),
                "downloaded": download_results["success"],
                "failed": download_results["failed"],
                "errors": download_results["errors"]
            }
        else:
            return {
                "new_files": len(all_new_files),
                "new_downloads": len(all_new_files),
                "new_lessons": len(new_lessons_created),
                "downloaded": 0,
                "failed": 0,
                "errors": []
            }
    
    def get_download_queue(self) -> List[Dict]:
        """Get all files in download queue"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM download_queue ORDER BY queued_date DESC")
        files = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': row[0],
                'name': row[1],
                'type': row[2],
                'lesson': row[3],
                'drive_folder': row[4],
                'queued_date': row[5],
                'status': row[6],
                'error_message': row[7]
            }
            for row in files
        ]
    
    def get_downloaded_files(self) -> List[Dict]:
        """Get all downloaded files"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM drive_files ORDER BY download_date DESC")
        files = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': row[0],
                'name': row[1],
                'type': row[2],
                'lesson': row[3],
                'drive_folder': row[4],
                'local_path': row[5],
                'download_date': row[6],
                'status': row[7],
                'file_size': row[8]
            }
            for row in files
        ]
    
    def start_automation(self):
        """Start the automation service"""
        print("🚀 Starting Google Drive automation service...")
        print(f"📅 Checking for new files every {self.config['processing']['check_interval_hours']} hour(s)")
        print(f"📥 Auto-download: {'Enabled' if self.config['processing']['auto_download'] else 'Disabled'}")
        print(f"🆕 Auto-create lessons: Enabled")
        
        # Run initial scan
        results = self.scan_and_download()
        print(f"✅ Initial scan complete: {results['new_files']} new files, {results['downloaded']} downloaded")
        if results.get('new_lessons', 0) > 0:
            print(f"🆕 {results['new_lessons']} new lesson(s) created")
        
        # Schedule regular scans
        import schedule
        schedule.every(self.config['processing']['check_interval_hours']).hours.do(self.scan_and_download)
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

if __name__ == "__main__":
    automation = DriveAutomation()
    automation.start_automation()
