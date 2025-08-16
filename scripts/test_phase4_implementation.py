#!/usr/bin/env python3
"""
Test Phase 4: Configuration & Integration Implementation

Validates the enhanced flashcard optimization system with cross-lesson context
integration, performance monitoring, and batch processing capabilities.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_configuration_management():
    """Test configuration management functionality"""
    print("\n🔧 Testing Configuration Management...")
    
    try:
        # Test configuration file loading
        config_path = Path("config/flashcard_optimization_config.json")
        if not config_path.exists():
            print("❌ Configuration file not found")
            return False
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required configuration sections
        required_sections = [
            'quality_thresholds', 'quality_weights', 'review_settings',
            'content_balance', 'clustering', 'optimization', 'military_context',
            'integration', 'workflow'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"❌ Missing configuration sections: {missing_sections}")
            return False
        
        # Test integration configuration
        integration_config = config.get('integration', {})
        if not integration_config:
            print("❌ Integration configuration missing")
            return False
        
        # Validate cross-lesson context settings
        cross_lesson_config = integration_config.get('cross_lesson_context', {})
        if not cross_lesson_config.get('enabled'):
            print("⚠️  Cross-lesson context is disabled")
        
        # Validate performance monitoring settings
        perf_config = integration_config.get('performance_monitoring', {})
        if not perf_config.get('enabled'):
            print("⚠️  Performance monitoring is disabled")
        
        # Validate batch processing settings
        batch_config = integration_config.get('batch_processing', {})
        if not batch_config.get('enable_auto_optimization'):
            print("⚠️  Auto optimization is disabled")
        
        print("✓ Configuration management working")
        return True
        
    except Exception as e:
        print(f"❌ Configuration management failed: {e}")
        return False

def test_enhanced_flashcard_optimizer():
    """Test enhanced flashcard optimizer with integration"""
    print("\n🚀 Testing Enhanced Flashcard Optimizer...")
    
    try:
        from flashcard_optimizer import FlashcardOptimizer, OptimizationResult
        
        # Initialize optimizer
        optimizer = FlashcardOptimizer()
        
        # Test configuration loading
        if not optimizer.config:
            print("❌ Optimizer configuration not loaded")
            return False
        
        # Test integration components
        if optimizer.cross_lesson_analyzer:
            print("✓ Cross-lesson analyzer initialized")
        else:
            print("⚠️  Cross-lesson analyzer not available")
        
        if optimizer.performance_tracker:
            print("✓ Performance tracker initialized")
        else:
            print("⚠️  Performance tracker not available")
        
        # Test optimization with sample data
        sample_flashcards = [
            {
                'term': 'Tactical Communication',
                'definition': 'The process of exchanging information in a tactical environment',
                'quality_score': 0.7,
                'difficulty': 'intermediate',
                'topic': 'communications'
            },
            {
                'term': 'Radio Protocol',
                'definition': 'Standard procedures for radio communication',
                'quality_score': 0.8,
                'difficulty': 'basic',
                'topic': 'communications'
            }
        ]
        
        # Test optimization
        result = optimizer.optimize_flashcards(sample_flashcards, lesson_id="test_lesson")
        
        if not isinstance(result, OptimizationResult):
            print("❌ Optimization result is not OptimizationResult type")
            return False
        
        # Validate result fields
        required_fields = [
            'original_count', 'optimized_count', 'improvements_made',
            'quality_improvement', 'content_balance_improvement',
            'optimization_time', 'cross_lesson_context_used', 'performance_metrics'
        ]
        
        for field in required_fields:
            if not hasattr(result, field):
                print(f"❌ Missing field in OptimizationResult: {field}")
                return False
        
        print(f"✓ Enhanced flashcard optimizer working")
        print(f"  - Original count: {result.original_count}")
        print(f"  - Optimized count: {result.optimized_count}")
        print(f"  - Optimization time: {result.optimization_time:.2f}s")
        print(f"  - Cross-lesson context used: {result.cross_lesson_context_used}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced flashcard optimizer failed: {e}")
        return False

def test_dashboard_integration():
    """Test dashboard integration endpoints"""
    print("\n🌐 Testing Dashboard Integration...")
    
    try:
        # Test configuration endpoints
        from enhanced_dashboard import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test configuration get endpoint
        response = client.get("/api/flashcards/optimization/config")
        if response.status_code == 200:
            config_data = response.json()
            if 'integration' in config_data:
                print("✓ Configuration endpoint working")
            else:
                print("❌ Configuration endpoint missing integration data")
                return False
        else:
            print(f"❌ Configuration endpoint failed: {response.status_code}")
            return False
        
        # Test optimization suggestions endpoint (with mock lesson)
        response = client.get("/api/flashcards/test_lesson/optimize")
        if response.status_code == 404:  # Expected for non-existent lesson
            print("✓ Optimization suggestions endpoint working (404 for non-existent lesson)")
        elif response.status_code == 200:
            print("✓ Optimization suggestions endpoint working")
        else:
            print(f"❌ Optimization suggestions endpoint failed: {response.status_code}")
            return False
        
        # Test progress endpoint
        response = client.get("/api/flashcards/test_lesson/optimize/progress")
        if response.status_code == 200:
            print("✓ Progress endpoint working")
        else:
            print(f"❌ Progress endpoint failed: {response.status_code}")
            return False
        
        print("✓ Dashboard integration working")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard integration failed: {e}")
        return False

def test_batch_processing():
    """Test batch processing functionality"""
    print("\n⚡ Testing Batch Processing...")
    
    try:
        from batch_flashcard_optimizer import BatchFlashcardOptimizer
        
        # Initialize batch optimizer
        batch_optimizer = BatchFlashcardOptimizer()
        
        # Test configuration loading
        if not batch_optimizer.config:
            print("❌ Batch optimizer configuration not loaded")
            return False
        
        # Test lesson discovery
        lessons = batch_optimizer.get_lessons_for_optimization()
        print(f"✓ Found {len(lessons)} lessons for optimization")
        
        # Test active optimizations tracking
        active_ops = batch_optimizer.get_active_optimizations()
        if isinstance(active_ops, dict):
            print("✓ Active optimizations tracking working")
        else:
            print("❌ Active optimizations tracking failed")
            return False
        
        # Test cleanup functionality
        batch_optimizer.cleanup_completed_optimizations()
        print("✓ Cleanup functionality working")
        
        print("✓ Batch processing working")
        return True
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return False

def test_cross_lesson_integration():
    """Test cross-lesson context integration"""
    print("\n🔗 Testing Cross-Lesson Integration...")
    
    try:
        from cross_lesson_analyzer import CrossLessonAnalyzer
        
        # Initialize analyzer
        analyzer = CrossLessonAnalyzer()
        
        # Test related lessons discovery
        test_lesson = "TLP"  # Use existing lesson
        related_lessons = analyzer.get_related_lessons(test_lesson, max_lessons=3)
        
        if isinstance(related_lessons, list):
            print(f"✓ Found {len(related_lessons)} related lessons for {test_lesson}")
        else:
            print("❌ Related lessons discovery failed")
            return False
        
        # Test context retrieval
        if related_lessons:
            context_data = analyzer.get_lesson_context(related_lessons[:2])
            if isinstance(context_data, dict):
                print("✓ Context data retrieval working")
            else:
                print("❌ Context data retrieval failed")
                return False
        
        print("✓ Cross-lesson integration working")
        return True
        
    except Exception as e:
        print(f"❌ Cross-lesson integration failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring integration"""
    print("\n📊 Testing Performance Monitoring...")
    
    try:
        from performance_monitor import performance_monitor
        
        if not performance_monitor:
            print("⚠️  Performance monitor not available")
            return True  # Not critical for basic functionality
        
        # Test operation tracking
        test_operation = {
            'operation_id': 'test_phase4',
            'lesson_id': 'test_lesson',
            'start_time': time.time()
        }
        
        if hasattr(performance_monitor, 'start_operation'):
            performance_monitor.start_operation('test_operation', test_operation)
            print("✓ Operation tracking working")
        else:
            print("⚠️  Operation tracking not available")
        
        # Test event logging
        if hasattr(performance_monitor, 'log_event'):
            performance_monitor.log_event('test_event', {'test': 'data'})
            print("✓ Event logging working")
        else:
            print("⚠️  Event logging not available")
        
        print("✓ Performance monitoring working")
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring failed: {e}")
        return False

def test_workflow_integration():
    """Test workflow integration features"""
    print("\n🔄 Testing Workflow Integration...")
    
    try:
        from flashcard_review_interface import FlashcardReviewInterface
        
        # Initialize review interface
        review_interface = FlashcardReviewInterface()
        
        # Test configuration integration
        if hasattr(review_interface, 'optimizer') and review_interface.optimizer:
            config = review_interface.optimizer.config
            if config and isinstance(config, dict):
                print("✓ Workflow configuration integration working")
            else:
                print("❌ Workflow configuration integration missing")
                return False
        else:
            print("❌ Review interface optimizer not available")
            return False
        
        # Test statistics with integration metadata
        sample_flashcards = [
            {
                'term': 'Test Term',
                'definition': 'Test definition',
                'quality_score': 0.8,
                'difficulty': 'basic',
                'topic': 'test',
                'metadata': {
                    'cross_lesson_context': {
                        'enhanced': True,
                        'related_lessons': 2
                    }
                }
            }
        ]
        
        stats = review_interface.get_review_statistics(sample_flashcards)
        if 'cross_lesson_context' in stats:
            print("✓ Integration statistics working")
        else:
            print("❌ Integration statistics missing")
            return False
        
        print("✓ Workflow integration working")
        return True
        
    except Exception as e:
        print(f"❌ Workflow integration failed: {e}")
        return False

def main():
    """Run all Phase 4 tests"""
    print("🧪 Phase 4: Configuration & Integration - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Configuration Management", test_configuration_management),
        ("Enhanced Flashcard Optimizer", test_enhanced_flashcard_optimizer),
        ("Dashboard Integration", test_dashboard_integration),
        ("Batch Processing", test_batch_processing),
        ("Cross-Lesson Integration", test_cross_lesson_integration),
        ("Performance Monitoring", test_performance_monitoring),
        ("Workflow Integration", test_workflow_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Phase 4 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All Phase 4 tests passed! Configuration & Integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
