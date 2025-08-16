#!/usr/bin/env python3
"""
Batch Flashcard Optimizer

Integrates flashcard optimization with the existing batch processing system,
providing automated optimization across multiple lessons with cross-lesson context.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import existing modules
try:
    from src.processing.flashcard_optimizer import FlashcardOptimizer
    from src.web.flashcard_review_interface import FlashcardReviewInterface
    from src.analysis.performance_monitor import performance_monitor
    from src.data.cross_lesson_analyzer import CrossLessonAnalyzer
    from src.data.model_manager import model_manager
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")

class BatchFlashcardOptimizer:
    """Batch processor for flashcard optimization across multiple lessons"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the batch optimizer"""
        self.config = config or self._load_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.optimizer = FlashcardOptimizer(self.config)
        self.cross_lesson_analyzer = None
        self.performance_tracker = None
        
        if CrossLessonAnalyzer:
            try:
                self.cross_lesson_analyzer = CrossLessonAnalyzer()
            except Exception as e:
                self.logger.warning(f"Could not initialize cross-lesson analyzer: {e}")
        
        if performance_monitor:
            self.performance_tracker = performance_monitor
        
        # Batch processing settings
        self.batch_config = self.config.get('integration', {}).get('batch_processing', {})
        self.max_concurrent = self.batch_config.get('max_concurrent_optimizations', 3)
        self.optimization_interval = self.batch_config.get('optimization_interval', 3600)
        self.priority_lessons = self.batch_config.get('priority_lessons', [])
        
        # Threading
        self.lock = threading.Lock()
        self.active_optimizations = {}
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            config_path = Path("config/flashcard_optimization_config.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load config file: {e}")
        
        return {}
    
    def get_lessons_for_optimization(self) -> List[str]:
        """Get list of lessons that need optimization"""
        try:
            lessons_dir = Path("lessons")
            if not lessons_dir.exists():
                return []
            
            lessons = []
            for lesson_dir in lessons_dir.iterdir():
                if lesson_dir.is_dir():
                    lesson_id = lesson_dir.name
                    
                    # Check if lesson has flashcards
                    output_dir = lesson_dir / "output"
                    if output_dir.exists():
                        flashcard_files = list(output_dir.glob("*flashcards*.json"))
                        if flashcard_files:
                            lessons.append(lesson_id)
            
            # Sort by priority
            priority_set = set(self.priority_lessons)
            lessons.sort(key=lambda x: (x not in priority_set, x))
            
            return lessons
            
        except Exception as e:
            self.logger.error(f"Error getting lessons for optimization: {e}")
            return []
    
    def optimize_lesson(self, lesson_id: str) -> Dict[str, Any]:
        """Optimize flashcards for a single lesson"""
        try:
            lesson_path = f"lessons/{lesson_id}"
            if not Path(lesson_path).exists():
                return {
                    'lesson_id': lesson_id,
                    'success': False,
                    'error': 'Lesson directory not found'
                }
            
            # Track optimization start
            optimization_id = f"batch_opt_{lesson_id}_{int(time.time())}"
            self.active_optimizations[optimization_id] = {
                'lesson_id': lesson_id,
                'start_time': time.time(),
                'status': 'running'
            }
            
            if self.performance_tracker:
                self.performance_tracker.start_operation('batch_flashcard_optimization', {
                    'optimization_id': optimization_id,
                    'lesson_id': lesson_id
                })
            
            # Load flashcards
            review_interface = FlashcardReviewInterface()
            review_data = review_interface.load_flashcards_for_review(lesson_path)
            flashcards = review_data.get("flashcards", [])
            
            if not flashcards:
                return {
                    'lesson_id': lesson_id,
                    'success': False,
                    'error': 'No flashcards found'
                }
            
            # Run optimization
            optimization_result = self.optimizer.optimize_flashcards(flashcards, lesson_id=lesson_id)
            
            # Save optimized flashcards
            success = review_interface.save_reviewed_flashcards(
                optimization_result, 
                lesson_path
            )
            
            # Update active optimizations
            with self.lock:
                if optimization_id in self.active_optimizations:
                    self.active_optimizations[optimization_id].update({
                        'status': 'completed',
                        'end_time': time.time(),
                        'success': success,
                        'result': {
                            'original_count': optimization_result.original_count,
                            'optimized_count': optimization_result.optimized_count,
                            'quality_improvement': optimization_result.quality_improvement,
                            'content_balance_improvement': optimization_result.content_balance_improvement,
                            'optimization_time': optimization_result.optimization_time,
                            'cross_lesson_context_used': optimization_result.cross_lesson_context_used
                        }
                    })
            
            # Track completion
            if self.performance_tracker:
                self.performance_tracker.end_operation('batch_flashcard_optimization', {
                    'optimization_id': optimization_id,
                    'success': success,
                    'optimization_result': optimization_result.performance_metrics
                })
                
                self.performance_tracker.log_event('batch_optimization_completed', {
                    'lesson_id': lesson_id,
                    'optimization_id': optimization_id,
                    'success': success,
                    'improvements': optimization_result.improvements_made
                })
            
            return {
                'lesson_id': lesson_id,
                'optimization_id': optimization_id,
                'success': success,
                'result': {
                    'original_count': optimization_result.original_count,
                    'optimized_count': optimization_result.optimized_count,
                    'quality_improvement': optimization_result.quality_improvement,
                    'content_balance_improvement': optimization_result.content_balance_improvement,
                    'optimization_time': optimization_result.optimization_time,
                    'cross_lesson_context_used': optimization_result.cross_lesson_context_used,
                    'improvements_made': optimization_result.improvements_made
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing lesson {lesson_id}: {e}")
            
            # Update active optimizations
            with self.lock:
                if optimization_id in self.active_optimizations:
                    self.active_optimizations[optimization_id].update({
                        'status': 'failed',
                        'end_time': time.time(),
                        'error': str(e)
                    })
            
            # Track error
            if self.performance_tracker:
                self.performance_tracker.log_event('batch_optimization_error', {
                    'lesson_id': lesson_id,
                    'error': str(e)
                })
            
            return {
                'lesson_id': lesson_id,
                'success': False,
                'error': str(e)
            }
    
    def run_batch_optimization(self, lesson_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run batch optimization on multiple lessons"""
        try:
            if lesson_ids is None:
                lesson_ids = self.get_lessons_for_optimization()
            
            if not lesson_ids:
                return {
                    'success': False,
                    'error': 'No lessons found for optimization'
                }
            
            self.logger.info(f"Starting batch optimization for {len(lesson_ids)} lessons")
            
            # Track batch start
            batch_id = f"batch_{int(time.time())}"
            if self.performance_tracker:
                self.performance_tracker.start_operation('batch_optimization_session', {
                    'batch_id': batch_id,
                    'total_lessons': len(lesson_ids)
                })
            
            results = []
            completed = 0
            failed = 0
            
            # Run optimizations with thread pool
            with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                # Submit all optimization tasks
                future_to_lesson = {
                    executor.submit(self.optimize_lesson, lesson_id): lesson_id 
                    for lesson_id in lesson_ids
                }
                
                # Process completed tasks
                for future in as_completed(future_to_lesson):
                    lesson_id = future_to_lesson[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result.get('success'):
                            completed += 1
                            self.logger.info(f"✓ Optimized lesson {lesson_id}")
                        else:
                            failed += 1
                            self.logger.error(f"✗ Failed to optimize lesson {lesson_id}: {result.get('error')}")
                    
                    except Exception as e:
                        failed += 1
                        self.logger.error(f"✗ Exception optimizing lesson {lesson_id}: {e}")
                        results.append({
                            'lesson_id': lesson_id,
                            'success': False,
                            'error': str(e)
                        })
            
            # Calculate batch statistics
            total_time = time.time()
            success_rate = completed / len(lesson_ids) if lesson_ids else 0
            
            batch_result = {
                'batch_id': batch_id,
                'total_lessons': len(lesson_ids),
                'completed': completed,
                'failed': failed,
                'success_rate': success_rate,
                'results': results,
                'total_time': total_time
            }
            
            # Track batch completion
            if self.performance_tracker:
                self.performance_tracker.end_operation('batch_optimization_session', batch_result)
                
                self.performance_tracker.log_event('batch_optimization_session_completed', {
                    'batch_id': batch_id,
                    'total_lessons': len(lesson_ids),
                    'success_rate': success_rate,
                    'completed': completed,
                    'failed': failed
                })
            
            self.logger.info(f"Batch optimization completed: {completed}/{len(lesson_ids)} lessons optimized successfully")
            
            return batch_result
            
        except Exception as e:
            self.logger.error(f"Error in batch optimization: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_active_optimizations(self) -> Dict[str, Any]:
        """Get currently active optimizations"""
        with self.lock:
            return self.active_optimizations.copy()
    
    def cleanup_completed_optimizations(self, max_age_hours: int = 24):
        """Clean up completed optimizations older than specified age"""
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            with self.lock:
                to_remove = []
                for opt_id, opt_data in self.active_optimizations.items():
                    if opt_data.get('status') in ['completed', 'failed']:
                        end_time = opt_data.get('end_time', 0)
                        if current_time - end_time > max_age_seconds:
                            to_remove.append(opt_id)
                
                for opt_id in to_remove:
                    del self.active_optimizations[opt_id]
            
            if to_remove:
                self.logger.info(f"Cleaned up {len(to_remove)} completed optimizations")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up optimizations: {e}")

def main():
    """Main function for batch optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch Flashcard Optimizer")
    parser.add_argument("--lessons", nargs="+", help="Specific lesson IDs to optimize")
    parser.add_argument("--all", action="store_true", help="Optimize all lessons")
    parser.add_argument("--priority", action="store_true", help="Only optimize priority lessons")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Maximum concurrent optimizations")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return
    
    # Initialize batch optimizer
    batch_optimizer = BatchFlashcardOptimizer(config)
    
    # Determine lessons to optimize
    lesson_ids = None
    if args.lessons:
        lesson_ids = args.lessons
    elif args.priority:
        lesson_ids = batch_optimizer.priority_lessons
    elif args.all:
        lesson_ids = batch_optimizer.get_lessons_for_optimization()
    else:
        print("Please specify --lessons, --all, or --priority")
        return
    
    if not lesson_ids:
        print("No lessons found for optimization")
        return
    
    # Override max concurrent if specified
    if args.max_concurrent:
        batch_optimizer.max_concurrent = args.max_concurrent
    
    # Run batch optimization
    print(f"Starting batch optimization for {len(lesson_ids)} lessons...")
    result = batch_optimizer.run_batch_optimization(lesson_ids)
    
    if result.get('success') is not False:
        print(f"\nBatch optimization completed:")
        print(f"  Total lessons: {result['total_lessons']}")
        print(f"  Completed: {result['completed']}")
        print(f"  Failed: {result['failed']}")
        print(f"  Success rate: {result['success_rate']:.1%}")
        print(f"  Total time: {result['total_time']:.2f}s")
    else:
        print(f"Batch optimization failed: {result.get('error')}")

if __name__ == "__main__":
    main()
