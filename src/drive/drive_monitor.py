import os
import json
import time
import schedule
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import sqlite3
from pathlib import Path
import re

class DriveMonitor:
    def __init__(self):
        self.config = self.load_config()
        self.service = self.authenticate_drive()
        self.db_path = "config/drive_monitor.db"
        self.init_database()
        
    def load_config(self):
        """Load configuration from config file"""
        with open("config/drive_config.json", "r") as f:
            return json.load(f)
    
    def authenticate_drive(self):
        """Authenticate with Google Drive API"""
        creds = Credentials.from_authorized_user_file("config/token.json", 
                                                     ['https://www.googleapis.com/auth/drive.readonly'])
        return build('drive', 'v3', credentials=creds)
    
    def init_database(self):
        """Initialize SQLite database for tracking files"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_files (
                file_id TEXT PRIMARY KEY,
                file_name TEXT,
                file_type TEXT,
                lesson_name TEXT,
                processed_date TEXT,
                status TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pending_files (
                file_id TEXT PRIMARY KEY,
                file_name TEXT,
                file_type TEXT,
                lesson_name TEXT,
                detected_date TEXT,
                status TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def detect_lesson_name(self, filename):
        """Intelligently detect lesson name from filename"""
        # Remove file extensions and common suffixes
        clean_name = re.sub(r'\.(pptx?|jpg|jpeg|m4a|mp3|pdf)$', '', filename, flags=re.IGNORECASE)
        clean_name = re.sub(r'\s*\(\d+\)\s*$', '', clean_name)  # Remove (2), (3), etc.
        
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
                return match.group(0)
        
        # If no pattern matches, return the cleaned filename
        return clean_name.strip()
    
    def categorize_file(self, mime_type):
        """Categorize file based on MIME type"""
        if mime_type in self.config['file_types']['audio']:
            return 'audio'
        elif mime_type in self.config['file_types']['presentations']:
            return 'presentation'
        elif mime_type in self.config['file_types']['notes']:
            return 'notes'
        else:
            return 'unknown'
    
    def scan_drive_folder(self, folder_id, folder_type):
        """Scan a specific Google Drive folder for new files"""
        try:
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
            new_files = []
            
            for file in files:
                file_id = file['id']
                file_name = file['name']
                mime_type = file['mimeType']
                
                # Check if file is already processed
                if not self.is_file_processed(file_id):
                    file_type = self.categorize_file(mime_type)
                    lesson_name = self.detect_lesson_name(file_name)
                    
                    if file_type != 'unknown':
                        new_files.append({
                            'id': file_id,
                            'name': file_name,
                            'type': file_type,
                            'lesson': lesson_name,
                            'mime_type': mime_type,
                            'folder_type': folder_type
                        })
            
            return new_files
            
        except Exception as e:
            print(f"Error scanning folder {folder_id}: {str(e)}")
            return []
    
    def is_file_processed(self, file_id):
        """Check if file has already been processed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT file_id FROM processed_files WHERE file_id = ?", (file_id,))
        result = cursor.fetchone()
        conn.close()
        return result is not None
    
    def add_pending_file(self, file_info):
        """Add file to pending processing queue"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO pending_files 
            (file_id, file_name, file_type, lesson_name, detected_date, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            file_info['id'],
            file_info['name'],
            file_info['type'],
            file_info['lesson'],
            datetime.now().isoformat(),
            'pending'
        ))
        
        conn.commit()
        conn.close()
    
    def scan_all_folders(self):
        """Scan all configured Google Drive folders"""
        print(f"\n[{datetime.now()}] Scanning Google Drive folders...")
        
        all_new_files = []
        
        # Scan personal folder (notes and audio)
        personal_files = self.scan_drive_folder(
            self.config['drive_folders']['personal_notes_audio'],
            'personal'
        )
        all_new_files.extend(personal_files)
        
        # Scan instructor folder (presentations)
        instructor_files = self.scan_drive_folder(
            self.config['drive_folders']['instructor_presentations'],
            'instructor'
        )
        all_new_files.extend(instructor_files)
        
        # Add new files to pending queue
        for file_info in all_new_files:
            self.add_pending_file(file_info)
            print(f"  New file detected: {file_info['name']} -> {file_info['lesson']} ({file_info['type']})")
        
        if all_new_files:
            self.send_notification_email(all_new_files)
        
        return all_new_files
    
    def send_notification_email(self, new_files):
        """Send email notification about new files"""
        if not self.config['processing']['email_notifications']['enabled']:
            return
        
        try:
            email = self.config['processing']['email_notifications']['email']
            
            # Create email content
            subject = f"New Files Detected - DriveToQuizlet ({len(new_files)} files)"
            
            body = f"""
            <h2>New Files Detected in Google Drive</h2>
            <p>The following {len(new_files)} new files have been detected and are ready for processing:</p>
            
            <table border="1" style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f0f0f0;">
                    <th>File Name</th>
                    <th>Lesson</th>
                    <th>Type</th>
                    <th>Source</th>
                </tr>
            """
            
            for file_info in new_files:
                body += f"""
                <tr>
                    <td>{file_info['name']}</td>
                    <td>{file_info['lesson']}</td>
                    <td>{file_info['type']}</td>
                    <td>{file_info['folder_type']}</td>
                </tr>
                """
            
            body += """
            </table>
            
            <p><strong>Next Steps:</strong></p>
            <ul>
                <li>Open the DriveToQuizlet web interface</li>
                <li>Review the pending files</li>
                <li>Approve processing for each lesson</li>
                <li>Download the generated TSV files for Quizlet</li>
            </ul>
            
            <p>Best regards,<br>DriveToQuizlet Automation System</p>
            """
            
            # Send email (you'll need to configure SMTP settings)
            # For now, just print the email content
            print(f"\nEmail notification would be sent to {email}")
            print(f"Subject: {subject}")
            print("Email content preview:")
            print(body[:500] + "..." if len(body) > 500 else body)
            
        except Exception as e:
            print(f"Error sending email notification: {str(e)}")
    
    def get_pending_files(self):
        """Get all pending files from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pending_files ORDER BY detected_date DESC")
        files = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': row[0],
                'name': row[1],
                'type': row[2],
                'lesson': row[3],
                'detected_date': row[4],
                'status': row[5]
            }
            for row in files
        ]
    
    def mark_file_processed(self, file_id):
        """Mark file as processed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Move from pending to processed
        cursor.execute("SELECT * FROM pending_files WHERE file_id = ?", (file_id,))
        pending_file = cursor.fetchone()
        
        if pending_file:
            cursor.execute('''
                INSERT INTO processed_files 
                (file_id, file_name, file_type, lesson_name, processed_date, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (pending_file[0], pending_file[1], pending_file[2], 
                  pending_file[3], datetime.now().isoformat(), 'completed'))
            
            cursor.execute("DELETE FROM pending_files WHERE file_id = ?", (file_id,))
        
        conn.commit()
        conn.close()
    
    def start_monitoring(self):
        """Start the monitoring service"""
        print("Starting Google Drive monitoring service...")
        print(f"Checking for new files every {self.config['processing']['check_interval_hours']} hour(s)")
        
        # Run initial scan
        self.scan_all_folders()
        
        # Schedule regular scans
        schedule.every(self.config['processing']['check_interval_hours']).hours.do(self.scan_all_folders)
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

if __name__ == "__main__":
    monitor = DriveMonitor()
    monitor.start_monitoring()

