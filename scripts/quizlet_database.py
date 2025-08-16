"""
Quizlet Database Management Module

This module handles the database schema and operations for Quizlet API integration,
including upload tracking, set management, and queue processing.
"""

import sqlite3
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuizletDatabase:
    """Database manager for Quizlet upload tracking and management."""
    
    def __init__(self, db_path: str = "config/quizlet_uploads.db"):
        """Initialize the Quizlet database manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.ensure_db_directory()
        self.init_database()
    
    def ensure_db_directory(self):
        """Ensure the database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            logger.info(f"Created database directory: {db_dir}")
    
    def init_database(self):
        """Initialize the database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create quizlet_sets table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS quizlet_sets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        lesson_name TEXT NOT NULL,
                        quizlet_set_id TEXT UNIQUE,
                        quizlet_set_url TEXT,
                        card_count INTEGER DEFAULT 0,
                        upload_status TEXT DEFAULT 'pending',
                        upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_sync DATETIME,
                        error_message TEXT,
                        set_title TEXT,
                        set_description TEXT,
                        visibility TEXT DEFAULT 'public',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create quizlet_upload_queue table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS quizlet_upload_queue (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        lesson_name TEXT NOT NULL,
                        tsv_file_path TEXT NOT NULL,
                        priority INTEGER DEFAULT 0,
                        status TEXT DEFAULT 'queued',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        processed_at DATETIME,
                        error_message TEXT,
                        retry_count INTEGER DEFAULT 0,
                        max_retries INTEGER DEFAULT 3,
                        next_retry DATETIME
                    )
                """)
                
                # Create quizlet_auth_tokens table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS quizlet_auth_tokens (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        access_token TEXT NOT NULL,
                        refresh_token TEXT NOT NULL,
                        expires_at DATETIME NOT NULL,
                        token_type TEXT DEFAULT 'Bearer',
                        scope TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create quizlet_upload_logs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS quizlet_upload_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        lesson_name TEXT NOT NULL,
                        action TEXT NOT NULL,
                        status TEXT NOT NULL,
                        details TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_quizlet_sets_lesson_name ON quizlet_sets(lesson_name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_quizlet_sets_status ON quizlet_sets(upload_status)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_upload_queue_status ON quizlet_upload_queue(status)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_upload_queue_priority ON quizlet_upload_queue(priority)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_auth_tokens_expires ON quizlet_auth_tokens(expires_at)")
                
                conn.commit()
                logger.info("Quizlet database initialized successfully")
                
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def add_quizlet_set(self, lesson_name: str, quizlet_set_id: str = None, 
                       quizlet_set_url: str = None, card_count: int = 0,
                       set_title: str = None, set_description: str = None,
                       visibility: str = "public") -> int:
        """Add a new Quizlet set record.
        
        Args:
            lesson_name: Name of the lesson
            quizlet_set_id: Quizlet set ID (if already created)
            quizlet_set_url: URL to the Quizlet set
            card_count: Number of cards in the set
            set_title: Title of the Quizlet set
            set_description: Description of the Quizlet set
            visibility: Set visibility (public/private)
            
        Returns:
            Database ID of the created record
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO quizlet_sets 
                    (lesson_name, quizlet_set_id, quizlet_set_url, card_count, 
                     set_title, set_description, visibility, upload_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (lesson_name, quizlet_set_id, quizlet_set_url, card_count,
                     set_title, set_description, visibility, 'pending'))
                
                conn.commit()
                set_id = cursor.lastrowid
                logger.info(f"Added Quizlet set record for lesson: {lesson_name}")
                return set_id
                
        except sqlite3.Error as e:
            logger.error(f"Error adding Quizlet set: {e}")
            raise
    
    def update_quizlet_set(self, set_id: int, **kwargs) -> bool:
        """Update a Quizlet set record.
        
        Args:
            set_id: Database ID of the set record
            **kwargs: Fields to update
            
        Returns:
            True if update was successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build dynamic update query
                update_fields = []
                values = []
                for key, value in kwargs.items():
                    if key in ['quizlet_set_id', 'quizlet_set_url', 'card_count', 
                              'upload_status', 'last_sync', 'error_message', 
                              'set_title', 'set_description', 'visibility']:
                        update_fields.append(f"{key} = ?")
                        values.append(value)
                
                if not update_fields:
                    return False
                
                update_fields.append("updated_at = ?")
                values.append(datetime.now())
                values.append(set_id)
                
                query = f"UPDATE quizlet_sets SET {', '.join(update_fields)} WHERE id = ?"
                cursor.execute(query, values)
                conn.commit()
                
                logger.info(f"Updated Quizlet set record ID: {set_id}")
                return cursor.rowcount > 0
                
        except sqlite3.Error as e:
            logger.error(f"Error updating Quizlet set: {e}")
            raise
    
    def get_quizlet_set(self, lesson_name: str) -> Optional[Dict]:
        """Get Quizlet set information for a lesson.
        
        Args:
            lesson_name: Name of the lesson
            
        Returns:
            Dictionary with set information or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM quizlet_sets 
                    WHERE lesson_name = ? 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """, (lesson_name,))
                
                row = cursor.fetchone()
                return dict(row) if row else None
                
        except sqlite3.Error as e:
            logger.error(f"Error getting Quizlet set: {e}")
            raise
    
    def add_upload_queue_item(self, lesson_name: str, tsv_file_path: str, 
                            priority: int = 0) -> int:
        """Add an item to the upload queue.
        
        Args:
            lesson_name: Name of the lesson
            tsv_file_path: Path to the TSV file
            priority: Upload priority (higher = more important)
            
        Returns:
            Database ID of the queue item
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO quizlet_upload_queue 
                    (lesson_name, tsv_file_path, priority, status)
                    VALUES (?, ?, ?, ?)
                """, (lesson_name, tsv_file_path, priority, 'queued'))
                
                conn.commit()
                queue_id = cursor.lastrowid
                logger.info(f"Added upload queue item for lesson: {lesson_name}")
                return queue_id
                
        except sqlite3.Error as e:
            logger.error(f"Error adding upload queue item: {e}")
            raise
    
    def get_upload_queue(self, status: str = None, limit: int = 50) -> List[Dict]:
        """Get items from the upload queue.
        
        Args:
            status: Filter by status (queued, processing, completed, failed)
            limit: Maximum number of items to return
            
        Returns:
            List of queue items
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = """
                    SELECT * FROM quizlet_upload_queue 
                    WHERE 1=1
                """
                params = []
                
                if status:
                    query += " AND status = ?"
                    params.append(status)
                
                query += " ORDER BY priority DESC, created_at ASC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except sqlite3.Error as e:
            logger.error(f"Error getting upload queue: {e}")
            raise
    
    def update_queue_item_status(self, queue_id: int, status: str, 
                                error_message: str = None) -> bool:
        """Update the status of a queue item.
        
        Args:
            queue_id: Database ID of the queue item
            status: New status
            error_message: Error message if status is 'failed'
            
        Returns:
            True if update was successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if status == 'processing':
                    cursor.execute("""
                        UPDATE quizlet_upload_queue 
                        SET status = ?, processed_at = ?, error_message = ?
                        WHERE id = ?
                    """, (status, datetime.now(), error_message, queue_id))
                else:
                    cursor.execute("""
                        UPDATE quizlet_upload_queue 
                        SET status = ?, error_message = ?
                        WHERE id = ?
                    """, (status, error_message, queue_id))
                
                conn.commit()
                logger.info(f"Updated queue item {queue_id} status to: {status}")
                return cursor.rowcount > 0
                
        except sqlite3.Error as e:
            logger.error(f"Error updating queue item status: {e}")
            raise
    
    def store_auth_tokens(self, access_token: str, refresh_token: str, 
                         expires_in: int, token_type: str = "Bearer", 
                         scope: str = None) -> bool:
        """Store OAuth2 tokens in the database.
        
        Args:
            access_token: OAuth2 access token
            refresh_token: OAuth2 refresh token
            expires_in: Token expiration time in seconds
            token_type: Token type (usually "Bearer")
            scope: OAuth2 scope
            
        Returns:
            True if tokens were stored successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clear existing tokens
                cursor.execute("DELETE FROM quizlet_auth_tokens")
                
                # Calculate expiration time
                expires_at = datetime.now() + timedelta(seconds=expires_in)
                
                # Store new tokens
                cursor.execute("""
                    INSERT INTO quizlet_auth_tokens 
                    (access_token, refresh_token, expires_at, token_type, scope)
                    VALUES (?, ?, ?, ?, ?)
                """, (access_token, refresh_token, expires_at, token_type, scope))
                
                conn.commit()
                logger.info("Stored new Quizlet auth tokens")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Error storing auth tokens: {e}")
            raise
    
    def get_auth_tokens(self) -> Optional[Dict]:
        """Get the current OAuth2 tokens.
        
        Returns:
            Dictionary with token information or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM quizlet_auth_tokens 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                
                row = cursor.fetchone()
                return dict(row) if row else None
                
        except sqlite3.Error as e:
            logger.error(f"Error getting auth tokens: {e}")
            raise
    
    def is_token_expired(self, token_data: Dict) -> bool:
        """Check if the access token is expired.
        
        Args:
            token_data: Token data from get_auth_tokens()
            
        Returns:
            True if token is expired or will expire soon
        """
        if not token_data:
            return True
        
        expires_at = datetime.fromisoformat(token_data['expires_at'])
        # Consider token expired if it expires within the next hour
        return expires_at <= datetime.now() + timedelta(hours=1)
    
    def log_upload_action(self, lesson_name: str, action: str, status: str, 
                         details: str = None):
        """Log an upload action for auditing.
        
        Args:
            lesson_name: Name of the lesson
            action: Action performed (upload, retry, delete, etc.)
            status: Status of the action (success, failed, pending)
            details: Additional details about the action
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO quizlet_upload_logs 
                    (lesson_name, action, status, details)
                    VALUES (?, ?, ?, ?)
                """, (lesson_name, action, status, details))
                
                conn.commit()
                logger.info(f"Logged upload action: {action} for {lesson_name}")
                
        except sqlite3.Error as e:
            logger.error(f"Error logging upload action: {e}")
            # Don't raise here as logging shouldn't break the main flow
    
    def get_upload_stats(self) -> Dict:
        """Get upload statistics.
        
        Returns:
            Dictionary with upload statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get set statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_sets,
                        SUM(CASE WHEN upload_status = 'completed' THEN 1 ELSE 0 END) as completed_sets,
                        SUM(CASE WHEN upload_status = 'failed' THEN 1 ELSE 0 END) as failed_sets,
                        SUM(CASE WHEN upload_status = 'pending' THEN 1 ELSE 0 END) as pending_sets,
                        SUM(card_count) as total_cards
                    FROM quizlet_sets
                """)
                set_stats = cursor.fetchone()
                
                # Get queue statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_queue_items,
                        SUM(CASE WHEN status = 'queued' THEN 1 ELSE 0 END) as queued_items,
                        SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as processing_items,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_items,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_items
                    FROM quizlet_upload_queue
                """)
                queue_stats = cursor.fetchone()
                
                return {
                    'sets': {
                        'total': set_stats[0] or 0,
                        'completed': set_stats[1] or 0,
                        'failed': set_stats[2] or 0,
                        'pending': set_stats[3] or 0,
                        'total_cards': set_stats[4] or 0
                    },
                    'queue': {
                        'total': queue_stats[0] or 0,
                        'queued': queue_stats[1] or 0,
                        'processing': queue_stats[2] or 0,
                        'completed': queue_stats[3] or 0,
                        'failed': queue_stats[4] or 0
                    }
                }
                
        except sqlite3.Error as e:
            logger.error(f"Error getting upload stats: {e}")
            raise


def init_quizlet_database():
    """Initialize the Quizlet database and return the database manager."""
    db = QuizletDatabase()
    return db


if __name__ == "__main__":
    # Test database initialization
    db = init_quizlet_database()
    print("Quizlet database initialized successfully!")
    
    # Test statistics
    stats = db.get_upload_stats()
    print("Upload statistics:", json.dumps(stats, indent=2))
