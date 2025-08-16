#!/usr/bin/env python3
"""
Performance Monitor for AI Model Loading and Processing

Tracks and reports performance metrics for model loading times,
processing speeds, and resource usage.
"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        """Initialize the performance monitor."""
        self.start_time = time.time()
        self.metrics = {
            "startup_time": 0,
            "model_loading_times": {},
            "processing_times": {},
            "memory_usage": [],
            "gpu_usage": [],
            "errors": []
        }
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start background monitoring of system resources."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
            self.monitor_thread.start()
            print("📊 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        print("📊 Performance monitoring stopped")
    
    def _monitor_resources(self):
        """Monitor system resources in background."""
        while self.monitoring:
            try:
                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics["memory_usage"].append({
                    "timestamp": time.time(),
                    "percent": memory.percent,
                    "used_gb": memory.used / (1024**3),
                    "available_gb": memory.available / (1024**3)
                })
                
                # GPU usage (if available)
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated(0) / (1024**3)
                        gpu_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                        self.metrics["gpu_usage"].append({
                            "timestamp": time.time(),
                            "allocated_gb": gpu_memory,
                            "reserved_gb": gpu_reserved
                        })
                except ImportError:
                    pass
                
                time.sleep(1)  # Sample every second
                
            except Exception as e:
                self.metrics["errors"].append({
                    "timestamp": time.time(),
                    "error": str(e)
                })
    
    def record_model_loading(self, model_name: str, load_time: float, success: bool = True):
        """Record model loading time."""
        self.metrics["model_loading_times"][model_name] = {
            "load_time": load_time,
            "success": success,
            "timestamp": time.time()
        }
        print(f"📈 {model_name} loaded in {load_time:.2f}s")
    
    def record_processing(self, task_name: str, processing_time: float, success: bool = True):
        """Record processing time for a task."""
        self.metrics["processing_times"][task_name] = {
            "processing_time": processing_time,
            "success": success,
            "timestamp": time.time()
        }
        print(f"📈 {task_name} completed in {processing_time:.2f}s")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "uptime": time.time() - self.start_time,
            "total_models": len(self.metrics["model_loading_times"]),
            "total_tasks": len(self.metrics["processing_times"]),
            "errors": len(self.metrics["errors"]),
            "metrics": self.metrics
        }
    
    def start_operation(self, operation_type: str, metadata: Dict[str, Any]):
        """Start tracking an operation."""
        operation_id = f"{operation_type}_{int(time.time())}"
        self.metrics["processing_times"][operation_id] = {
            "type": operation_type,
            "start_time": time.time(),
            "metadata": metadata,
            "status": "running"
        }
        return operation_id
    
    def end_operation(self, operation_type: str, result: Dict[str, Any]):
        """End tracking an operation."""
        operation_id = f"{operation_type}_{int(time.time())}"
        if operation_id in self.metrics["processing_times"]:
            self.metrics["processing_times"][operation_id].update({
                "end_time": time.time(),
                "duration": time.time() - self.metrics["processing_times"][operation_id]["start_time"],
                "result": result,
                "status": "completed"
            })
    
    def log_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log an event."""
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "data": event_data
        }
        if "events" not in self.metrics:
            self.metrics["events"] = []
        self.metrics["events"].append(event)
    
    def get_active_operations(self) -> List[Dict[str, Any]]:
        """Get currently active operations."""
        active_ops = []
        for op_id, op_data in self.metrics["processing_times"].items():
            if op_data.get("status") == "running":
                active_ops.append({
                    "operation_id": op_id,
                    **op_data
                })
        return active_ops
    
    def get_optimization_metrics(self, lesson_id: str) -> Dict[str, Any]:
        """Get optimization metrics for a specific lesson."""
        lesson_metrics = []
        for op_id, op_data in self.metrics["processing_times"].items():
            if (op_data.get("type") == "flashcard_optimization" and 
                op_data.get("metadata", {}).get("lesson_id") == lesson_id):
                lesson_metrics.append({
                    "operation_id": op_id,
                    **op_data
                })
        return {
            "lesson_id": lesson_id,
            "optimizations": lesson_metrics,
            "total_optimizations": len(lesson_metrics)
        }
    
    def save_report(self, filename: str = None):
        """Save detailed performance report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_summary(),
            "detailed_metrics": self.metrics
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Performance report saved: {filename}")
        return filename
    
    def print_summary(self):
        """Print a formatted summary of performance metrics."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("📊 PERFORMANCE SUMMARY")
        print("="*60)
        print(f"⏱️  Total Runtime: {summary['total_runtime']:.2f}s")
        print(f"🚀 Startup Time: {summary['startup_time']:.2f}s")
        
        if summary['model_loading_summary']:
            print("\n🤖 Model Loading Times:")
            for model, load_time in summary['model_loading_summary'].items():
                print(f"  - {model}: {load_time:.2f}s")
        
        if summary['processing_summary']:
            print("\n⚙️  Processing Times:")
            for task, proc_time in summary['processing_summary'].items():
                print(f"  - {task}: {proc_time:.2f}s")
        
        print(f"\n💾 Resource Usage:")
        print(f"  - Avg Memory: {summary['resource_usage']['avg_memory_percent']:.1f}%")
        print(f"  - Peak Memory: {summary['resource_usage']['peak_memory_percent']:.1f}%")
        print(f"  - Avg GPU Memory: {summary['resource_usage']['avg_gpu_memory_gb']:.2f}GB")
        print(f"  - Peak GPU Memory: {summary['resource_usage']['peak_gpu_memory_gb']:.2f}GB")
        
        if summary['errors'] > 0:
            print(f"\n⚠️  Errors: {summary['errors']}")
        
        print("="*60)

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def track_model_loading(model_name: str):
    """Decorator to track model loading time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                load_time = time.time() - start_time
                performance_monitor.record_model_loading(model_name, load_time, True)
                return result
            except Exception as e:
                load_time = time.time() - start_time
                performance_monitor.record_model_loading(model_name, load_time, False)
                raise
        return wrapper
    return decorator

def track_processing(task_name: str):
    """Decorator to track processing time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                processing_time = time.time() - start_time
                performance_monitor.record_processing(task_name, processing_time, True)
                return result
            except Exception as e:
                processing_time = time.time() - start_time
                performance_monitor.record_processing(task_name, processing_time, False)
                raise
        return wrapper
    return decorator
