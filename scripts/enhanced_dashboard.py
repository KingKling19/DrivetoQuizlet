#!/usr/bin/env python3
"""
Enhanced DriveToQuizlet Dashboard

A comprehensive web interface that provides important information and allows
execution of key commands for the DriveToQuizlet military training system.
"""

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import json
import os
import sqlite3
import subprocess
import threading
import time
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio

# Import DriveToQuizlet modules
try:
    from drive_automation import DriveAutomation
    from model_manager import model_manager
    from performance_monitor import performance_monitor
    from optimized_file_operations import file_ops
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")

app = FastAPI(
    title="DriveToQuizlet Enhanced Dashboard", 
    version="2.0.0",
    description="Comprehensive interface for military training lesson processing"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global instances
automation = None
system_status = {
    "startup_time": datetime.now(),
    "last_scan": None,
    "last_process": None,
    "active_processes": [],
    "errors": []
}

class SystemMonitor:
    """Monitor system resources and status"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU info if available
            gpu_info = {}
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_info = {
                        "available": True,
                        "device_count": torch.cuda.device_count(),
                        "current_device": torch.cuda.current_device(),
                        "device_name": torch.cuda.get_device_name(0),
                        "memory_allocated": torch.cuda.memory_allocated(0) / (1024**3),
                        "memory_reserved": torch.cuda.memory_reserved(0) / (1024**3)
                    }
                else:
                    gpu_info = {"available": False}
            except ImportError:
                gpu_info = {"available": False, "error": "PyTorch not available"}
            
            return {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "percent": memory.percent
                },
                "disk": {
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "percent": (disk.used / disk.total) * 100
                },
                "gpu": gpu_info,
                "uptime": (datetime.now() - system_status["startup_time"]).total_seconds()
            }
        except Exception as e:
            return {"error": str(e)}

@app.on_event("startup")
async def startup_event():
    """Initialize the dashboard on startup"""
    global automation
    try:
        automation = DriveAutomation()
        print("✓ DriveAutomation initialized")
    except Exception as e:
        print(f"⚠️  DriveAutomation initialization failed: {e}")
        system_status["errors"].append(f"DriveAutomation: {e}")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("enhanced_dashboard.html", {"request": request})

# ============================================================================
# SYSTEM INFORMATION ENDPOINTS
# ============================================================================

@app.get("/api/system/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        # System resources
        system_info = SystemMonitor.get_system_info()
        
        # Model manager status
        model_status = {}
        try:
            model_status = model_manager.get_cache_status()
        except:
            model_status = {"error": "Model manager not available"}
        
        # Performance monitor status
        perf_status = {}
        try:
            perf_status = performance_monitor.get_summary()
        except:
            perf_status = {"error": "Performance monitor not available"}
        
        # Configuration status
        config_status = {}
        try:
            config_path = Path("config/drive_config.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                config_status = {
                    "configured": True,
                    "folders": len(config.get("drive_folders", {})),
                    "auto_download": config.get("processing", {}).get("auto_download", False),
                    "check_interval": config.get("processing", {}).get("check_interval_hours", 1)
                }
            else:
                config_status = {"configured": False}
        except Exception as e:
            config_status = {"error": str(e)}
        
        return {
            "system": system_info,
            "models": model_status,
            "performance": perf_status,
            "config": config_status,
            "dashboard": system_status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/health")
async def get_system_health():
    """Get system health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }
    
    # Check critical components
    checks = health_status["checks"]
    
    # Check if automation is working
    try:
        if automation:
            checks["automation"] = {"status": "ok", "message": "DriveAutomation initialized"}
        else:
            checks["automation"] = {"status": "error", "message": "DriveAutomation not initialized"}
            health_status["status"] = "degraded"
    except Exception as e:
        checks["automation"] = {"status": "error", "message": str(e)}
        health_status["status"] = "degraded"
    
    # Check configuration files
    config_files = ["config/drive_config.json", "config/token.json"]
    for config_file in config_files:
        if Path(config_file).exists():
            checks[f"config_{config_file.split('/')[-1]}"] = {"status": "ok", "message": "File exists"}
        else:
            checks[f"config_{config_file.split('/')[-1]}"] = {"status": "error", "message": "File missing"}
            health_status["status"] = "degraded"
    
    # Check database files
    db_files = ["config/drive_automation.db", "config/drive_monitor.db"]
    for db_file in db_files:
        if Path(db_file).exists():
            checks[f"db_{db_file.split('/')[-1]}"] = {"status": "ok", "message": "Database exists"}
        else:
            checks[f"db_{db_file.split('/')[-1]}"] = {"status": "warning", "message": "Database missing"}
    
    # Check lessons directory
    if Path("lessons").exists():
        checks["lessons_dir"] = {"status": "ok", "message": "Lessons directory exists"}
    else:
        checks["lessons_dir"] = {"status": "warning", "message": "Lessons directory missing"}
    
    return health_status

# ============================================================================
# DRIVE AUTOMATION ENDPOINTS
# ============================================================================

@app.get("/api/drive/status")
async def get_drive_status():
    """Get Google Drive automation status"""
    if not automation:
        raise HTTPException(status_code=500, detail="DriveAutomation not initialized")
    
    try:
        # Get pending files
        pending_files = automation.get_download_queue()
        
        # Get downloaded files
        downloaded_files = automation.get_downloaded_files()
        
        # Get configuration
        config = automation.config
        
        return {
            "pending_files": len(pending_files),
            "downloaded_files": len(downloaded_files),
            "config": {
                "folders": config.get("drive_folders", {}),
                "auto_download": config.get("processing", {}).get("auto_download", False),
                "check_interval": config.get("processing", {}).get("check_interval_hours", 1)
            },
            "last_scan": system_status.get("last_scan"),
            "status": "active" if automation else "inactive"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/drive/scan")
async def scan_drive(background_tasks: BackgroundTasks):
    """Trigger a Google Drive scan"""
    if not automation:
        raise HTTPException(status_code=500, detail="DriveAutomation not initialized")
    
    def run_scan():
        try:
            system_status["last_scan"] = datetime.now()
            results = automation.scan_and_download()
            return results
        except Exception as e:
            system_status["errors"].append(f"Scan error: {e}")
            raise e
    
    background_tasks.add_task(run_scan)
    
    return {"status": "scanning", "message": "Drive scan started in background"}

@app.get("/api/drive/pending-files")
async def get_pending_files():
    """Get all pending files for processing"""
    if not automation:
        raise HTTPException(status_code=500, detail="DriveAutomation not initialized")
    
    try:
        files = automation.get_download_queue()
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/drive/downloaded-files")
async def get_downloaded_files():
    """Get all downloaded files"""
    if not automation:
        raise HTTPException(status_code=500, detail="DriveAutomation not initialized")
    
    try:
        files = automation.get_downloaded_files()
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/drive/process-queue")
async def process_download_queue(background_tasks: BackgroundTasks):
    """Process the download queue"""
    if not automation:
        raise HTTPException(status_code=500, detail="DriveAutomation not initialized")
    
    def run_processing():
        try:
            system_status["last_process"] = datetime.now()
            results = automation.process_download_queue()
            return results
        except Exception as e:
            system_status["errors"].append(f"Processing error: {e}")
            raise e
    
    background_tasks.add_task(run_processing)
    
    return {"status": "processing", "message": "Download queue processing started"}

# ============================================================================
# LESSON MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/api/lessons")
async def get_lessons():
    """Get all lessons with detailed status"""
    lessons_dir = Path("lessons")
    lessons = []
    
    if lessons_dir.exists():
        for lesson_dir in lessons_dir.iterdir():
            if lesson_dir.is_dir():
                # Count files in each subdirectory
                presentations = list((lesson_dir / "presentations").glob("*")) if (lesson_dir / "presentations").exists() else []
                notes = list((lesson_dir / "notes").glob("*")) if (lesson_dir / "notes").exists() else []
                audio = list((lesson_dir / "audio").glob("*")) if (lesson_dir / "audio").exists() else []
                processed = list((lesson_dir / "processed").glob("*")) if (lesson_dir / "processed").exists() else []
                output = list((lesson_dir / "output").glob("*.tsv")) if (lesson_dir / "output").exists() else []
                
                # Get file sizes
                total_size = sum(f.stat().st_size for f in presentations + notes + audio + processed + output if f.is_file())
                
                lesson_info = {
                    "name": lesson_dir.name,
                    "path": str(lesson_dir),
                    "files": {
                        "presentations": len(presentations),
                        "notes": len(notes),
                        "audio": len(audio),
                        "processed": len(processed),
                        "output": len(output)
                    },
                    "total_size_mb": total_size / (1024 * 1024),
                    "has_output": len(output) > 0,
                    "last_modified": datetime.fromtimestamp(lesson_dir.stat().st_mtime).isoformat(),
                    "status": "ready" if len(output) > 0 else "processing" if len(processed) > 0 else "pending"
                }
                lessons.append(lesson_info)
    
    return {"lessons": lessons}

@app.post("/api/lessons/{lesson_name}/process")
async def process_lesson(lesson_name: str, background_tasks: BackgroundTasks):
    """Process a specific lesson"""
    lesson_path = f"lessons/{lesson_name}"
    if not os.path.exists(lesson_path):
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    def run_lesson_processing():
        try:
            system_status["active_processes"].append(f"lesson_{lesson_name}")
            
            # Try optimized processor first
            result = subprocess.run(
                ["python", "scripts/process_lesson_optimized.py", lesson_name],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                # Fallback to regular processor
                result = subprocess.run(
                    ["python", "scripts/process_lesson.py", lesson_path],
                    capture_output=True,
                    text=True,
                    cwd=os.getcwd(),
                    timeout=300
                )
            
            if result.returncode == 0:
                system_status["last_process"] = datetime.now()
                return {"status": "success", "message": f"Lesson {lesson_name} processed successfully"}
            else:
                error_msg = result.stderr or result.stdout
                system_status["errors"].append(f"Lesson processing error: {error_msg}")
                return {"status": "error", "message": error_msg}
                
        except subprocess.TimeoutExpired:
            error_msg = f"Lesson processing timed out after 5 minutes"
            system_status["errors"].append(error_msg)
            return {"status": "error", "message": error_msg}
        except Exception as e:
            error_msg = f"Lesson processing failed: {str(e)}"
            system_status["errors"].append(error_msg)
            return {"status": "error", "message": error_msg}
        finally:
            if f"lesson_{lesson_name}" in system_status["active_processes"]:
                system_status["active_processes"].remove(f"lesson_{lesson_name}")
    
    background_tasks.add_task(run_lesson_processing)
    
    return {"status": "processing", "message": f"Lesson {lesson_name} processing started"}

@app.post("/api/lessons/{lesson_name}/organize")
async def organize_lesson(lesson_name: str):
    """Organize files for a lesson"""
    try:
        result = subprocess.run(
            ["python", "scripts/organize_lessons.py", "organize", "--lesson-name", lesson_name],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            return {"status": "success", "message": f"Lesson {lesson_name} organized successfully"}
        else:
            return {"status": "error", "message": result.stderr}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/lessons/{lesson_name}/download")
async def download_lesson_output(lesson_name: str):
    """Download TSV file for a lesson"""
    lesson_path = Path(f"lessons/{lesson_name}/output")
    if not lesson_path.exists():
        raise HTTPException(status_code=404, detail="Lesson output not found")
    
    # Find TSV files
    tsv_files = list(lesson_path.glob("*.tsv"))
    if not tsv_files:
        raise HTTPException(status_code=404, detail="No TSV files found")
    
    # Return the most recent TSV file
    latest_tsv = max(tsv_files, key=lambda f: f.stat().st_mtime)
    
    return FileResponse(
        path=str(latest_tsv),
        filename=f"{lesson_name}_flashcards.tsv",
        media_type="text/tab-separated-values"
    )

# ============================================================================
# COMMAND EXECUTION ENDPOINTS
# ============================================================================

@app.post("/api/commands/scan-drive")
async def execute_scan_drive():
    """Execute drive scan command"""
    return await scan_drive(BackgroundTasks())

@app.post("/api/commands/process-all-lessons")
async def execute_process_all_lessons(background_tasks: BackgroundTasks):
    """Execute batch processing of all lessons"""
    def run_batch_processing():
        try:
            system_status["active_processes"].append("batch_processing")
            
            result = subprocess.run(
                ["python", "scripts/batch_process_lessons.py"],
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                system_status["last_process"] = datetime.now()
                return {"status": "success", "message": "All lessons processed successfully"}
            else:
                error_msg = result.stderr or result.stdout
                system_status["errors"].append(f"Batch processing error: {error_msg}")
                return {"status": "error", "message": error_msg}
                
        except subprocess.TimeoutExpired:
            error_msg = "Batch processing timed out after 10 minutes"
            system_status["errors"].append(error_msg)
            return {"status": "error", "message": error_msg}
        except Exception as e:
            error_msg = f"Batch processing failed: {str(e)}"
            system_status["errors"].append(error_msg)
            return {"status": "error", "message": error_msg}
        finally:
            if "batch_processing" in system_status["active_processes"]:
                system_status["active_processes"].remove("batch_processing")
    
    background_tasks.add_task(run_batch_processing)
    
    return {"status": "processing", "message": "Batch processing started"}

@app.post("/api/commands/test-connection")
async def execute_test_connection():
    """Test Google Drive connection"""
    try:
        result = subprocess.run(
            ["python", "scripts/drive_cli.py", "test"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            return {"status": "success", "message": "Connection test successful", "output": result.stdout}
        else:
            return {"status": "error", "message": "Connection test failed", "output": result.stderr}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/commands/clear-cache")
async def execute_clear_cache():
    """Clear model cache"""
    try:
        if hasattr(model_manager, 'clear_cache'):
            model_manager.clear_cache()
            return {"status": "success", "message": "Model cache cleared"}
        else:
            return {"status": "error", "message": "Model manager not available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/commands/performance-test")
async def execute_performance_test():
    """Run performance test"""
    try:
        result = subprocess.run(
            ["python", "scripts/test_performance.py"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        return {
            "status": "success" if result.returncode == 0 else "error",
            "output": result.stdout,
            "error": result.stderr
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# LOGS AND MONITORING ENDPOINTS
# ============================================================================

@app.get("/api/logs/errors")
async def get_error_logs():
    """Get recent error logs"""
    return {
        "errors": system_status["errors"][-50:],  # Last 50 errors
        "total_errors": len(system_status["errors"])
    }

@app.get("/api/logs/active-processes")
async def get_active_processes():
    """Get currently active processes"""
    return {
        "active_processes": system_status["active_processes"],
        "count": len(system_status["active_processes"])
    }

@app.get("/api/logs/performance")
async def get_performance_logs():
    """Get performance monitoring data"""
    try:
        if hasattr(performance_monitor, 'get_summary'):
            return performance_monitor.get_summary()
        else:
            return {"error": "Performance monitor not available"}
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# FLASHCARD REVIEW INTERFACE ENDPOINTS
# ============================================================================

@app.get("/api/flashcards/{lesson_id}/optimize")
async def get_flashcard_optimization_suggestions(lesson_id: str):
    """Get optimization suggestions for flashcards with enhanced integration"""
    try:
        from flashcard_review_interface import FlashcardReviewInterface
        
        lesson_path = f"lessons/{lesson_id}"
        if not os.path.exists(lesson_path):
            raise HTTPException(status_code=404, detail=f"Lesson {lesson_id} not found")
        
        review_interface = FlashcardReviewInterface()
        review_data = review_interface.load_flashcards_for_review(lesson_path)
        
        # Get cross-lesson context information if available
        cross_lesson_info = {}
        try:
            from cross_lesson_analyzer import CrossLessonAnalyzer
            analyzer = CrossLessonAnalyzer()
            related_lessons = analyzer.get_related_lessons(lesson_id, max_lessons=5)
            if related_lessons:
                cross_lesson_info = {
                    'related_lessons': related_lessons,
                    'context_available': True,
                    'enhancement_enabled': True
                }
        except Exception as e:
            cross_lesson_info = {
                'context_available': False,
                'error': str(e)
            }
        
        # Get performance metrics if available
        performance_metrics = {}
        try:
            if hasattr(performance_monitor, 'get_optimization_metrics'):
                performance_metrics = performance_monitor.get_optimization_metrics(lesson_id)
        except Exception as e:
            performance_metrics = {'error': str(e)}
        
        return {
            "lesson_id": lesson_id,
            "suggestions": review_data.get("suggestions", []),
            "summary": review_data.get("summary", {}),
            "total_flashcards": review_data.get("total_flashcards", 0),
            "cross_lesson_context": cross_lesson_info,
            "performance_metrics": performance_metrics,
            "optimization_config": review_interface.optimizer.config.get('integration', {})
        }
    except HTTPException:
        # Re-raise HTTP exceptions (like 404) as-is
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/flashcards/{lesson_id}/optimize")
async def apply_flashcard_optimization(lesson_id: str, operations: List[Dict[str, Any]]):
    """Apply optimization operations to flashcards with enhanced tracking"""
    try:
        from flashcard_review_interface import FlashcardReviewInterface
        
        lesson_path = f"lessons/{lesson_id}"
        if not os.path.exists(lesson_path):
            raise HTTPException(status_code=404, detail=f"Lesson {lesson_id} not found")
        
        review_interface = FlashcardReviewInterface()
        review_data = review_interface.load_flashcards_for_review(lesson_path)
        
        # Track optimization start
        optimization_id = f"opt_{lesson_id}_{int(time.time())}"
        if hasattr(performance_monitor, 'start_operation'):
            performance_monitor.start_operation('flashcard_optimization', {
                'optimization_id': optimization_id,
                'lesson_id': lesson_id,
                'operations_count': len(operations)
            })
        
        # Apply bulk operations
        optimized_flashcards = review_interface.apply_bulk_operations(
            review_data.get("flashcards", []), 
            operations
        )
        
        # Run full optimization with cross-lesson context
        optimization_result = review_interface.optimizer.optimize_flashcards(
            optimized_flashcards, 
            lesson_id=lesson_id
        )
        
        # Save optimized flashcards
        success = review_interface.save_reviewed_flashcards(optimization_result, lesson_path)
        
        # Track optimization completion
        if hasattr(performance_monitor, 'end_operation'):
            performance_monitor.end_operation('flashcard_optimization', {
                'optimization_id': optimization_id,
                'success': success,
                'optimization_result': {
                    'original_count': optimization_result.original_count,
                    'optimized_count': optimization_result.optimized_count,
                    'quality_improvement': optimization_result.quality_improvement,
                    'content_balance_improvement': optimization_result.content_balance_improvement,
                    'optimization_time': optimization_result.optimization_time,
                    'cross_lesson_context_used': optimization_result.cross_lesson_context_used
                }
            })
        
        return {
            "lesson_id": lesson_id,
            "optimization_id": optimization_id,
            "success": success,
            "optimized_count": optimization_result.optimized_count,
            "operations_applied": len(operations),
            "optimization_result": {
                "original_count": optimization_result.original_count,
                "quality_improvement": optimization_result.quality_improvement,
                "content_balance_improvement": optimization_result.content_balance_improvement,
                "optimization_time": optimization_result.optimization_time,
                "cross_lesson_context_used": optimization_result.cross_lesson_context_used,
                "improvements_made": optimization_result.improvements_made
            }
        }
    except Exception as e:
        # Track optimization error
        if hasattr(performance_monitor, 'log_event'):
            performance_monitor.log_event('flashcard_optimization_error', {
                'lesson_id': lesson_id,
                'error': str(e)
            })
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/flashcards/{lesson_id}/optimize/progress")
async def get_optimization_progress(lesson_id: str):
    """Get real-time optimization progress"""
    try:
        # Check if optimization is in progress
        if hasattr(performance_monitor, 'get_active_operations'):
            active_ops = performance_monitor.get_active_operations()
            optimization_ops = [op for op in active_ops if op.get('type') == 'flashcard_optimization' and op.get('lesson_id') == lesson_id]
            
            if optimization_ops:
                return {
                    "lesson_id": lesson_id,
                    "in_progress": True,
                    "operations": optimization_ops
                }
        
        return {
            "lesson_id": lesson_id,
            "in_progress": False
        }
    except Exception as e:
        return {
            "lesson_id": lesson_id,
            "in_progress": False,
            "error": str(e)
        }

@app.get("/api/flashcards/{lesson_id}/review")
async def load_flashcards_for_review(lesson_id: str):
    """Load flashcards for manual review with enhanced metadata"""
    try:
        from flashcard_review_interface import FlashcardReviewInterface
        
        lesson_path = f"lessons/{lesson_id}"
        if not os.path.exists(lesson_path):
            raise HTTPException(status_code=404, detail=f"Lesson {lesson_id} not found")
        
        review_interface = FlashcardReviewInterface()
        review_data = review_interface.load_flashcards_for_review(lesson_path)
        
        # Add integration metadata
        integration_metadata = {
            'cross_lesson_context_enabled': review_interface.optimizer.integration_config.get('cross_lesson_context', {}).get('enabled', False),
            'performance_monitoring_enabled': review_interface.optimizer.integration_config.get('performance_monitoring', {}).get('enabled', False),
            'auto_optimization_enabled': review_interface.optimizer.integration_config.get('batch_processing', {}).get('enable_auto_optimization', False)
        }
        
        return {
            "lesson_id": lesson_id,
            "flashcards": review_data.get("flashcards", []),
            "summary": review_data.get("summary", {}),
            "suggestions": review_data.get("suggestions", []),
            "clusters": review_data.get("clusters", {}),
            "loaded_at": review_data.get("loaded_at"),
            "integration_metadata": integration_metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/flashcards/{lesson_id}/review")
async def save_reviewed_flashcards(lesson_id: str, flashcards: List[Dict[str, Any]]):
    """Save reviewed flashcards with enhanced tracking"""
    try:
        from flashcard_review_interface import FlashcardReviewInterface
        
        lesson_path = f"lessons/{lesson_id}"
        if not os.path.exists(lesson_path):
            raise HTTPException(status_code=404, detail=f"Lesson {lesson_id} not found")
        
        review_interface = FlashcardReviewInterface()
        success = review_interface.save_reviewed_flashcards(flashcards, lesson_path)
        
        # Track review completion
        if hasattr(performance_monitor, 'log_event'):
            performance_monitor.log_event('flashcard_review_completed', {
                'lesson_id': lesson_id,
                'flashcards_reviewed': len(flashcards),
                'success': success
            })
        
        return {
            "lesson_id": lesson_id,
            "success": success,
            "saved_count": len(flashcards)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/flashcards/{lesson_id}/clusters")
async def get_flashcard_clusters(lesson_id: str):
    """Get clustering analysis for flashcards with enhanced metadata"""
    try:
        from flashcard_review_interface import FlashcardReviewInterface
        
        lesson_path = f"lessons/{lesson_id}"
        if not os.path.exists(lesson_path):
            raise HTTPException(status_code=404, detail=f"Lesson {lesson_id} not found")
        
        review_interface = FlashcardReviewInterface()
        review_data = review_interface.load_flashcards_for_review(lesson_path)
        
        clusters = review_data.get("clusters", {})
        cluster_summaries = {}
        
        if hasattr(review_interface, 'clusterer') and review_interface.clusterer:
            cluster_summaries = review_interface.clusterer.generate_cluster_summaries(clusters)
        
        return {
            "lesson_id": lesson_id,
            "clusters": clusters,
            "cluster_summaries": cluster_summaries,
            "clustering_config": review_interface.optimizer.config.get('clustering', {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/flashcards/{lesson_id}/statistics")
async def get_flashcard_statistics(lesson_id: str):
    """Get comprehensive flashcard statistics with integration metrics"""
    try:
        from flashcard_review_interface import FlashcardReviewInterface
        
        lesson_path = f"lessons/{lesson_id}"
        if not os.path.exists(lesson_path):
            raise HTTPException(status_code=404, detail=f"Lesson {lesson_id} not found")
        
        review_interface = FlashcardReviewInterface()
        review_data = review_interface.load_flashcards_for_review(lesson_path)
        
        statistics = review_interface.get_review_statistics(review_data.get("flashcards", []))
        
        # Add integration statistics
        integration_stats = {}
        try:
            if hasattr(performance_monitor, 'get_optimization_metrics'):
                integration_stats['optimization_metrics'] = performance_monitor.get_optimization_metrics(lesson_id)
            
            # Get cross-lesson context statistics
            from cross_lesson_analyzer import CrossLessonAnalyzer
            analyzer = CrossLessonAnalyzer()
            related_lessons = analyzer.get_related_lessons(lesson_id, max_lessons=10)
            integration_stats['cross_lesson_context'] = {
                'related_lessons_count': len(related_lessons),
                'context_available': len(related_lessons) > 0
            }
        except Exception as e:
            integration_stats['error'] = str(e)
        
        return {
            "lesson_id": lesson_id,
            "statistics": statistics,
            "integration_statistics": integration_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/flashcards/optimization/config")
async def get_optimization_config():
    """Get current optimization configuration"""
    try:
        config_path = Path("config/flashcard_optimization_config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        else:
            return {"error": "Configuration file not found"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/flashcards/optimization/config")
async def update_optimization_config(config_update: Dict[str, Any]):
    """Update optimization configuration"""
    try:
        config_path = Path("config/flashcard_optimization_config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update configuration
            for key, value in config_update.items():
                if key in config:
                    config[key].update(value)
                else:
                    config[key] = value
            
            # Save updated configuration
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            return {"success": True, "message": "Configuration updated successfully"}
        else:
            return {"error": "Configuration file not found"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/flashcards/{lesson_id}/review", response_class=HTMLResponse)
async def flashcard_review_interface(request: Request, lesson_id: str):
    """Serve the flashcard review interface"""
    try:
        lesson_path = f"lessons/{lesson_id}"
        if not os.path.exists(lesson_path):
            raise HTTPException(status_code=404, detail=f"Lesson {lesson_id} not found")
        
        return templates.TemplateResponse("flashcard_review.html", {
            "request": request,
            "lesson_id": lesson_id
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.get("/api/utils/disk-speed")
async def get_disk_speed():
    """Test disk I/O speed"""
    try:
        if hasattr(file_ops, 'get_disk_speed_test'):
            speed_test = file_ops.get_disk_speed_test()
            return speed_test
        else:
            return {"error": "File operations not available"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/utils/file-stats")
async def get_file_stats():
    """Get file system statistics"""
    try:
        lessons_dir = Path("lessons")
        total_files = 0
        total_size = 0
        
        if lessons_dir.exists():
            for file_path in lessons_dir.rglob("*"):
                if file_path.is_file():
                    total_files += 1
                    total_size += file_path.stat().st_size
        
        return {
            "total_files": total_files,
            "total_size_mb": total_size / (1024 * 1024),
            "total_size_gb": total_size / (1024 * 1024 * 1024)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_dashboard:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
