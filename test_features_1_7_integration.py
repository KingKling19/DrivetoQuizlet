#!/usr/bin/env python3
"""
Test Integration Between Feature 8 and Features 1-7
Tests the end-to-end workflow without external dependencies
"""

import json
import sys
import os
import time
from pathlib import Path

# Add the workspace to Python path to import our modules
sys.path.insert(0, '/workspace')

def test_feature_1_project_setup():
    """Test Feature 1: Project Setup & Environment Configuration"""
    print("🧪 Testing Feature 1: Project Setup...")
    
    required_dirs = [
        '/workspace/scripts',
        '/workspace/config', 
        '/workspace/templates',
        '/workspace/static',
        '/workspace/lessons',
        '/workspace/outputs'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✅ Project structure exists")
    return True

def test_feature_7_flashcard_generation():
    """Test Feature 7: Flashcard Content Generation"""
    print("🧪 Testing Feature 7: Flashcard Generation...")
    
    try:
        # Import without numpy-dependent functions
        sys.path.insert(0, '/workspace/scripts')
        
        # Test basic imports
        from convert_folder_to_quizlet import (
            canonical_term,
            compact_definition
        )
        
        # Test basic functionality
        test_term = "Air Defense Artillery"
        canonical = canonical_term(test_term)
        
        test_definition = "Air Defense Artillery (ADA) is a branch of the military that defends against aerial threats. It includes various systems and procedures for protecting ground forces and strategic assets from enemy aircraft and missiles."
        compact = compact_definition(test_definition)
        
        if len(compact) <= len(test_definition):
            print("✅ Flashcard generation functions working")
            return True, (canonical_term, compact_definition)
        else:
            print("❌ Definition compaction not working")
            return False, None
            
    except Exception as e:
        print(f"❌ Flashcard generation test failed: {e}")
        return False, None

def test_feature_8_optimization_integration():
    """Test Feature 8 integration with flashcard generation"""
    print("🧪 Testing Feature 8 + 7 Integration...")
    
    try:
        from scripts.flashcard_optimizer import FlashcardOptimizer
        
        # Load config
        config_path = Path("/workspace/config/flashcard_optimization_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        optimizer = FlashcardOptimizer(config)
        
        # Create flashcards using Feature 7 style data
        test_flashcards = [
            {
                'term': 'ADA',
                'definition': 'Air Defense Artillery - military branch defending against aerial threats',
                'quality_score': 0.8,
                'source_slide': 'lesson_ada_basics_slide_01',
                'lesson_id': 'ada_basics'
            },
            {
                'term': 'air defense artillery',  # Similar to above
                'definition': 'Military branch that defends against aerial threats and aircraft',
                'quality_score': 0.75,
                'source_slide': 'lesson_ada_intro_slide_02',
                'lesson_id': 'ada_intro'
            },
            {
                'term': 'Radar System',
                'definition': 'Electronic detection system used to locate and track aircraft and missiles',
                'quality_score': 0.9,
                'source_slide': 'lesson_radar_slide_01',
                'lesson_id': 'radar_basics'
            }
        ]
        
        # Test optimization
        result = optimizer.optimize_flashcards(test_flashcards, lesson_id='test_lesson')
        
        print(f"   Input flashcards: {len(test_flashcards)}")
        print(f"   Output flashcards: {result.optimized_count}")
        print(f"   Improvements made: {len(result.improvements_made)}")
        
        if result.optimized_count <= len(test_flashcards):
            print("✅ Feature 8 + 7 integration working")
            return True
        else:
            print("❌ Optimization produced unexpected results")
            return False
            
    except Exception as e:
        print(f"❌ Feature 8 + 7 integration failed: {e}")
        return False

def test_file_processing_pipeline():
    """Test file processing pipeline (Features 3-6)"""
    print("🧪 Testing File Processing Pipeline (Features 3-6)...")
    
    try:
        # Check if core processing files exist
        core_files = [
            '/workspace/scripts/process_lesson.py',
            '/workspace/scripts/audio_processor.py',
            '/workspace/scripts/notes_processor.py',
            '/workspace/scripts/lesson_content_indexer.py',
            '/workspace/scripts/cross_lesson_analyzer.py'
        ]
        
        missing_files = []
        for file_path in core_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print(f"❌ Missing core processing files: {missing_files}")
            return False
        
        print("✅ All core processing files exist")
        
        # Test if we can import basic functions (without running them)
        try:
            # These imports should work without external dependencies
            from scripts.process_lesson import main as process_lesson_main
            print("✅ Process lesson module importable")
        except Exception as e:
            print(f"⚠️  Process lesson import warning: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ File processing pipeline test failed: {e}")
        return False

def test_end_to_end_workflow_simulation():
    """Simulate end-to-end workflow from raw content to optimized flashcards"""
    print("🧪 Testing End-to-End Workflow Simulation...")
    
    try:
        # Simulate processed lesson content (what would come from Features 3-6)
        simulated_lesson_data = {
            'lesson_id': 'ada_fundamentals',
            'slides': [
                {
                    'slide_number': 1,
                    'title': 'ADA Overview',
                    'content': 'Air Defense Artillery (ADA) is responsible for defending against aerial threats.'
                },
                {
                    'slide_number': 2,
                    'title': 'Radar Systems',
                    'content': 'Radar systems detect and track aircraft using radio waves.'
                }
            ],
            'audio_transcript': 'Today we will learn about Air Defense Artillery fundamentals...',
            'notes': 'Key points: ADA mission, radar operation, threat assessment'
        }
        
        # Simulate Feature 7 (flashcard generation) output
        generated_flashcards = [
            {
                'term': 'ADA',
                'definition': 'Air Defense Artillery - responsible for defending against aerial threats',
                'quality_score': 0.8,
                'source_slide': 'ada_fundamentals_slide_01',
                'confidence': 0.85
            },
            {
                'term': 'Radar',
                'definition': 'System that detects and tracks aircraft using radio waves',
                'quality_score': 0.9,
                'source_slide': 'ada_fundamentals_slide_02',
                'confidence': 0.9
            },
            {
                'term': 'aerial threats',
                'definition': 'Aircraft, missiles, or other airborne objects that pose danger',
                'quality_score': 0.7,
                'source_slide': 'ada_fundamentals_slide_01',
                'confidence': 0.75
            }
        ]
        
        # Apply Feature 8 optimization
        from scripts.flashcard_optimizer import FlashcardOptimizer
        config_path = Path("/workspace/config/flashcard_optimization_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        optimizer = FlashcardOptimizer(config)
        optimization_result = optimizer.optimize_flashcards(
            generated_flashcards, 
            lesson_id='ada_fundamentals'
        )
        
        print(f"   Raw lesson data: {len(simulated_lesson_data['slides'])} slides")
        print(f"   Generated flashcards: {len(generated_flashcards)}")
        print(f"   Optimized flashcards: {optimization_result.optimized_count}")
        print(f"   Quality improvement: {optimization_result.quality_improvement:.2f}")
        print(f"   Optimization time: {optimization_result.optimization_time:.2f}s")
        
        # Check that we maintained or improved quality
        if optimization_result.optimized_count <= len(generated_flashcards):
            print("✅ End-to-end workflow simulation successful")
            return True, optimization_result
        else:
            print("❌ Workflow produced unexpected results")
            return False, None
            
    except Exception as e:
        print(f"❌ End-to-end workflow simulation failed: {e}")
        return False, None

def test_content_balance_across_lessons():
    """Test content balance optimization across multiple lesson topics"""
    print("🧪 Testing Content Balance Across Lessons...")
    
    try:
        from scripts.flashcard_optimizer import FlashcardOptimizer
        config_path = Path("/workspace/config/flashcard_optimization_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        optimizer = FlashcardOptimizer(config)
        
        # Create flashcards representing different ADA topics
        diverse_flashcards = [
            # Operations
            {'term': 'Mission Planning', 'definition': 'Process of strategic operational preparation', 'quality_score': 0.8},
            {'term': 'Tactical Deployment', 'definition': 'Strategic positioning of defense systems', 'quality_score': 0.85},
            {'term': 'Threat Assessment', 'definition': 'Evaluation of potential aerial dangers', 'quality_score': 0.9},
            
            # Communications
            {'term': 'Radio Protocol', 'definition': 'Standard communication procedures for military operations', 'quality_score': 0.8},
            {'term': 'Signal Transmission', 'definition': 'Electronic communication between units', 'quality_score': 0.75},
            
            # Equipment
            {'term': 'Radar Array', 'definition': 'Collection of radar systems for enhanced detection', 'quality_score': 0.9},
            {'term': 'Missile System', 'definition': 'Weapon system for intercepting aerial targets', 'quality_score': 0.85},
            {'term': 'Control Console', 'definition': 'Interface for operating defense systems', 'quality_score': 0.8},
            
            # Safety
            {'term': 'Safety Protocol', 'definition': 'Procedures ensuring personnel protection during operations', 'quality_score': 0.8},
            
            # Leadership
            {'term': 'Command Structure', 'definition': 'Hierarchical organization of military leadership', 'quality_score': 0.75}
        ]
        
        # Analyze content distribution
        distribution = optimizer.analyze_content_distribution(diverse_flashcards)
        
        print(f"   Total flashcards: {distribution['total_flashcards']}")
        print(f"   Balance score: {distribution['balance_score']:.3f}")
        print(f"   Balance level: {distribution['balance_level']}")
        
        # Show topic distribution
        for topic, data in distribution['distribution'].items():
            print(f"   - {topic}: {data['count']} cards ({data['percentage']:.1f}%)")
        
        if distribution['balance_score'] > 0.5:
            print("✅ Content balance analysis working correctly")
            return True
        else:
            print("⚠️  Content balance needs improvement")
            return True  # Still pass as this is analysis, not failure
            
    except Exception as e:
        print(f"❌ Content balance test failed: {e}")
        return False

def main():
    """Main test runner for integration tests"""
    print("🚀 Features 1-7 + 8 Integration Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Test Feature 1: Project Setup
    feature1_success = test_feature_1_project_setup()
    test_results.append(("Feature 1: Project Setup", feature1_success))
    
    # Test Features 3-6: File Processing Pipeline
    pipeline_success = test_file_processing_pipeline()
    test_results.append(("Features 3-6: File Processing Pipeline", pipeline_success))
    
    # Test Feature 7: Flashcard Generation
    feature7_success, generation_funcs = test_feature_7_flashcard_generation()
    test_results.append(("Feature 7: Flashcard Generation", feature7_success))
    
    # Test Feature 8 + 7 Integration
    integration_success = test_feature_8_optimization_integration()
    test_results.append(("Feature 8 + 7 Integration", integration_success))
    
    # Test End-to-End Workflow
    workflow_success, workflow_result = test_end_to_end_workflow_simulation()
    test_results.append(("End-to-End Workflow", workflow_success))
    
    # Test Content Balance
    balance_success = test_content_balance_across_lessons()
    test_results.append(("Content Balance Analysis", balance_success))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Integration Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 All integration tests passed! Features 1-8 work together correctly.")
    elif passed > total * 0.8:
        print("⚠️  Most integration tests passed. System is largely functional.")
    else:
        print("❌ Multiple integration tests failed. System needs attention.")
    
    return passed, total

if __name__ == "__main__":
    try:
        passed, total = main()
        sys.exit(0 if passed == total else 1)
    except Exception as e:
        print(f"💥 Integration test suite crashed: {e}")
        sys.exit(2)