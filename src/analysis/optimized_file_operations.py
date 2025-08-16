#!/usr/bin/env python3
"""
Optimized File Operations with Progress Indicators

Provides efficient file copying and moving operations with:
- Progress indicators for large files
- Better error handling
- Performance monitoring
- Background operations
"""

import os
import shutil
import time
import threading
from pathlib import Path
from typing import Callable, Optional, Dict, Any
import psutil

class OptimizedFileOperations:
    """Optimized file operations with progress tracking."""
    
    def __init__(self):
        self.operations = []
        self.current_operation = None
    
    def copy_file_with_progress(self, src: Path, dst: Path, 
                               progress_callback: Optional[Callable] = None) -> bool:
        """Copy a file with progress indication."""
        try:
            # Get file size for progress calculation
            file_size = src.stat().st_size
            print(f"📁 Copying {src.name} ({file_size / (1024*1024):.1f} MB)...")
            
            # For large files (>10MB), use chunked copying
            if file_size > 10 * 1024 * 1024:  # 10MB threshold
                return self._copy_large_file(src, dst, file_size, progress_callback)
            else:
                return self._copy_small_file(src, dst, progress_callback)
                
        except Exception as e:
            print(f"✗ Copy failed: {e}")
            return False
    
    def _copy_small_file(self, src: Path, dst: Path, 
                        progress_callback: Optional[Callable] = None) -> bool:
        """Copy small files using shutil.copy2."""
        try:
            start_time = time.time()
            shutil.copy2(src, dst)
            copy_time = time.time() - start_time
            
            if progress_callback:
                progress_callback(100, 100, copy_time)
            
            print(f"✓ Copied in {copy_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"✗ Small file copy failed: {e}")
            return False
    
    def _copy_large_file(self, src: Path, dst: Path, file_size: int,
                        progress_callback: Optional[Callable] = None) -> bool:
        """Copy large files in chunks with progress."""
        try:
            chunk_size = 1024 * 1024  # 1MB chunks
            copied_bytes = 0
            start_time = time.time()
            
            with open(src, 'rb') as fsrc:
                with open(dst, 'wb') as fdst:
                    while True:
                        chunk = fsrc.read(chunk_size)
                        if not chunk:
                            break
                        
                        fdst.write(chunk)
                        copied_bytes += len(chunk)
                        
                        # Calculate progress
                        progress = (copied_bytes / file_size) * 100
                        elapsed_time = time.time() - start_time
                        
                        # Update progress every 5% or every 5MB
                        if (int(progress) % 5 == 0 and progress > 0) or copied_bytes % (5 * 1024 * 1024) == 0:
                            print(f"  📊 {progress:.1f}% ({copied_bytes / (1024*1024):.1f} MB)")
                        
                        if progress_callback:
                            progress_callback(copied_bytes, file_size, elapsed_time)
            
            total_time = time.time() - start_time
            speed = file_size / (1024 * 1024 * total_time)  # MB/s
            
            print(f"✓ Copied in {total_time:.2f}s ({speed:.1f} MB/s)")
            return True
            
        except Exception as e:
            print(f"✗ Large file copy failed: {e}")
            return False
    
    def copy_multiple_files(self, file_mappings: Dict[Path, Path], 
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Copy multiple files with overall progress."""
        results = {
            "successful": [],
            "failed": [],
            "total_size": 0,
            "copied_size": 0,
            "start_time": time.time()
        }
        
        # Calculate total size
        for src, dst in file_mappings.items():
            if src.exists():
                results["total_size"] += src.stat().st_size
        
        print(f"📦 Copying {len(file_mappings)} files ({results['total_size'] / (1024*1024):.1f} MB total)")
        
        for i, (src, dst) in enumerate(file_mappings.items(), 1):
            print(f"\n[{i}/{len(file_mappings)}] Processing: {src.name}")
            
            if not src.exists():
                print(f"⚠️  Source file not found: {src}")
                results["failed"].append(str(src))
                continue
            
            # Ensure destination directory exists
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            if self.copy_file_with_progress(src, dst, progress_callback):
                results["successful"].append(str(src))
                results["copied_size"] += src.stat().st_size
            else:
                results["failed"].append(str(src))
        
        total_time = time.time() - results["start_time"]
        results["total_time"] = total_time
        
        # Print summary
        print(f"\n📊 Copy Summary:")
        print(f"  ✓ Successful: {len(results['successful'])}")
        print(f"  ✗ Failed: {len(results['failed'])}")
        print(f"  ⏱️  Total time: {total_time:.2f}s")
        print(f"  📁 Total copied: {results['copied_size'] / (1024*1024):.1f} MB")
        
        return results
    
    def background_copy(self, src: Path, dst: Path, 
                       completion_callback: Optional[Callable] = None) -> threading.Thread:
        """Copy file in background thread."""
        def copy_worker():
            try:
                success = self.copy_file_with_progress(src, dst)
                if completion_callback:
                    completion_callback(success, src, dst)
            except Exception as e:
                print(f"✗ Background copy failed: {e}")
                if completion_callback:
                    completion_callback(False, src, dst)
        
        thread = threading.Thread(target=copy_worker, daemon=True)
        thread.start()
        return thread
    
    def get_disk_speed_test(self, test_dir: Path = None) -> Dict[str, float]:
        """Test disk I/O speed for performance estimation."""
        if test_dir is None:
            test_dir = Path.cwd()
        
        print("🔍 Testing disk I/O speed...")
        
        # Create a test file
        test_file = test_dir / "speed_test.tmp"
        test_size = 10 * 1024 * 1024  # 10MB
        
        try:
            # Write test
            start_time = time.time()
            with open(test_file, 'wb') as f:
                f.write(b'0' * test_size)
            write_time = time.time() - start_time
            write_speed = test_size / (1024 * 1024 * write_time)  # MB/s
            
            # Read test
            start_time = time.time()
            with open(test_file, 'rb') as f:
                f.read()
            read_time = time.time() - start_time
            read_speed = test_size / (1024 * 1024 * read_time)  # MB/s
            
            # Cleanup
            test_file.unlink()
            
            print(f"✓ Write speed: {write_speed:.1f} MB/s")
            print(f"✓ Read speed: {read_speed:.1f} MB/s")
            
            return {
                "write_speed_mbps": write_speed,
                "read_speed_mbps": read_speed,
                "write_time": write_time,
                "read_time": read_time
            }
            
        except Exception as e:
            print(f"✗ Speed test failed: {e}")
            return {"write_speed_mbps": 0, "read_speed_mbps": 0}
    
    def estimate_copy_time(self, file_size: int, speed_mbps: float = None) -> float:
        """Estimate copy time based on file size and disk speed."""
        if speed_mbps is None:
            # Default conservative estimate
            speed_mbps = 50.0  # 50 MB/s
        
        estimated_time = file_size / (1024 * 1024 * speed_mbps)
        return estimated_time

# Global instance
file_ops = OptimizedFileOperations()

def copy_lesson_files_optimized(lesson_name: str, base_dir: Path = Path("downloads")) -> Dict[str, Any]:
    """Optimized version of lesson file copying with progress indicators."""
    print(f"🚀 Optimized file copying for lesson: {lesson_name}")
    
    # Test disk speed first
    speed_test = file_ops.get_disk_speed_test()
    
    # Create lesson structure
    lesson_dir = base_dir / "lessons" / lesson_name
    dirs = {
        "presentations_dir": lesson_dir / "presentations",
        "notes_dir": lesson_dir / "notes",
        "audio_dir": lesson_dir / "audio",
        "processed_dir": lesson_dir / "processed",
        "output_dir": lesson_dir / "output"
    }
    
    # Create directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Find and map files
    file_mappings = {}
    
    # PowerPoint files
    for pptx_file in base_dir.rglob("*.pptx"):
        if lesson_name.lower() in pptx_file.name.lower():
            dest = dirs["presentations_dir"] / pptx_file.name
            if not dest.exists():
                file_mappings[pptx_file] = dest
    
    # Quizlet files
    for quizlet_file in base_dir.rglob("*.quizlet.json"):
        if lesson_name.lower() in quizlet_file.name.lower():
            dest = dirs["processed_dir"] / quizlet_file.name
            if not dest.exists():
                file_mappings[quizlet_file] = dest
    
    # Note images
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        for note_file in base_dir.rglob(ext):
            if lesson_name.lower() in note_file.name.lower():
                dest = dirs["notes_dir"] / note_file.name
                if not dest.exists():
                    file_mappings[note_file] = dest
    
    # Audio files
    for audio_file in base_dir.rglob("*.m4a"):
        if lesson_name.lower() in audio_file.name.lower():
            dest = dirs["audio_dir"] / audio_file.name
            if not dest.exists():
                file_mappings[audio_file] = dest
    
    # Copy files with progress
    if file_mappings:
        results = file_ops.copy_multiple_files(file_mappings)
        results["speed_test"] = speed_test
        return results
    else:
        print("ℹ️  No files found to copy")
        return {"successful": [], "failed": [], "total_size": 0, "copied_size": 0}

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        lesson_name = sys.argv[1]
        results = copy_lesson_files_optimized(lesson_name)
        print(f"\n✅ Copy operation completed!")
    else:
        print("Usage: python optimized_file_operations.py <lesson_name>")
