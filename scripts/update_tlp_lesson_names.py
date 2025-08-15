#!/usr/bin/env python3
"""
Script to update TLP lesson names in the database to consolidate variations.
"""

import sqlite3
from pathlib import Path
import shutil

def update_tlp_lesson_names():
    """Update database to consolidate TLP lesson names"""
    db_path = "config/drive_automation.db"
    
    if not Path(db_path).exists():
        print("❌ Database not found. Run drive automation first to create it.")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Update drive_files table
    cursor.execute("""
        UPDATE drive_files 
        SET lesson_name = 'TLP' 
        WHERE lesson_name LIKE '%TLP%Overview%combined%' 
        OR lesson_name LIKE '%TLP%Overview%'
    """)
    
    files_updated = cursor.rowcount
    print(f"📝 Updated {files_updated} records in drive_files table")
    
    # Update download_queue table
    cursor.execute("""
        UPDATE download_queue 
        SET lesson_name = 'TLP' 
        WHERE lesson_name LIKE '%TLP%Overview%combined%' 
        OR lesson_name LIKE '%TLP%Overview%'
    """)
    
    queue_updated = cursor.rowcount
    print(f"📝 Updated {queue_updated} records in download_queue table")
    
    conn.commit()
    conn.close()
    
    print(f"✅ Database updated successfully!")
    print(f"   - {files_updated} files updated")
    print(f"   - {queue_updated} queue items updated")

def consolidate_tlp_folders():
    """Consolidate TLP folders if they still exist"""
    lessons_dir = Path("lessons")
    
    if not lessons_dir.exists():
        print("❌ Lessons directory not found")
        return
    
    # Check if TLP Overview combined folder still exists
    tlp_overview_dir = lessons_dir / "TLP Overview combined"
    tlp_dir = lessons_dir / "TLP"
    
    if tlp_overview_dir.exists() and tlp_dir.exists():
        print("🔄 Consolidating TLP folders...")
        
        # Move any remaining files from TLP Overview combined to TLP
        for item in tlp_overview_dir.rglob("*"):
            if item.is_file():
                # Calculate relative path
                rel_path = item.relative_to(tlp_overview_dir)
                target_path = tlp_dir / rel_path
                
                # Create target directory if it doesn't exist
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Move file
                shutil.move(str(item), str(target_path))
                print(f"   📁 Moved: {item.name} -> {target_path}")
        
        # Remove the empty TLP Overview combined folder
        shutil.rmtree(tlp_overview_dir)
        print(f"   🗑️  Removed: {tlp_overview_dir}")
    
    print("✅ Folder consolidation complete!")

if __name__ == "__main__":
    print("🔄 Updating TLP lesson names in database...")
    update_tlp_lesson_names()
    
    print("\n🔄 Consolidating TLP folders...")
    consolidate_tlp_folders()
    
    print("\n✅ TLP lesson consolidation complete!")


