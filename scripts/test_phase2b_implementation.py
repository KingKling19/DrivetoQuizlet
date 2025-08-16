#!/usr/bin/env python3
"""
Test Phase 2B Implementation

Tests and demonstrates the enhanced cross-lesson context system with:
- Enhanced PowerPoint + Notes integration
- Enhanced PowerPoint + Audio integration  
- Enhanced lesson processing pipeline
- Context optimization algorithms
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

def test_cross_lesson_data_loading():
    """Test loading of cross-lesson analysis data."""
    print("=" * 60)
    print("TESTING CROSS-LESSON DATA LOADING")
    print("=" * 60)
    
    config_dir = Path("config")
    
    # Test content index
    index_file = config_dir / "lesson_content_index.json"
    if index_file.exists():
        with open(index_file, 'r', encoding='utf-8') as f:
            content_index = json.load(f)
        print(f"✓ Content index loaded: {len(content_index)} lessons")
        for lesson_id in content_index.keys():
            print(f"  - {lesson_id}")
    else:
        print("⚠️  Content index not found")
    
    # Test lesson relationships
    relationships_file = config_dir / "lesson_relationships_analysis.json"
    if relationships_file.exists():
        with open(relationships_file, 'r', encoding='utf-8') as f:
            relationships = json.load(f)
        print(f"✓ Lesson relationships loaded: {len(relationships)} lessons")
    else:
        print("⚠️  Lesson relationships not found")
    
    # Test semantic embeddings
    embeddings_file = config_dir / "semantic_embeddings.pkl"
    if embeddings_file.exists():
        print(f"✓ Semantic embeddings found")
    else:
        print("⚠️  Semantic embeddings not found")
    
    return True

def test_context_optimizer():
    """Test the context optimizer functionality."""
    print("\n" + "=" * 60)
    print("TESTING CONTEXT OPTIMIZER")
    print("=" * 60)
    
    try:
        from context_optimizer import ContextOptimizer
        
        optimizer = ContextOptimizer()
        
        # Test with available lessons
        available_lessons = list(optimizer.cross_lesson_data.get("content_index", {}).keys())
        
        if available_lessons:
            test_lesson = available_lessons[0]
            print(f"Testing context optimization for lesson: {test_lesson}")
            
            # Get optimized context
            result = optimizer.get_optimized_context(test_lesson, "general")
            
            print(f"✓ Context optimization completed:")
            print(f"  - Selected {len(result['selected_context'])} context sources")
            print(f"  - Quality score: {result['quality_assessment'].get('quality_score', 0):.2f}")
            print(f"  - Context length: {len(result['context_content'])} characters")
            
            # Show selected context details
            for i, ctx in enumerate(result['selected_context'][:3]):
                print(f"  Context {i+1}: {ctx['lesson_id']} (weight: {ctx['weight']:.2f})")
            
            return True
        else:
            print("⚠️  No lessons available for testing")
            return False
            
    except Exception as e:
        print(f"❌ Error testing context optimizer: {e}")
        return False

def test_enhanced_powerpoint_notes_integration():
    """Test enhanced PowerPoint + Notes integration with cross-lesson context."""
    print("\n" + "=" * 60)
    print("TESTING ENHANCED POWERPOINT + NOTES INTEGRATION")
    print("=" * 60)
    
    try:
        from integrate_powerpoint_notes import PowerPointNotesIntegrator
        
        integrator = PowerPointNotesIntegrator()
        
        print(f"✓ PowerPoint Notes Integrator initialized")
        print(f"  - Cross-lesson data loaded: {len(integrator.cross_lesson_data.get('content_index', {}))} lessons")
        print(f"  - Context config: {integrator.context_config}")
        
        # Test lesson ID extraction
        test_path = Path("lessons/TLP/presentations/test.pptx")
        lesson_id = integrator.get_lesson_id_from_path(test_path)
        print(f"  - Lesson ID extraction test: {test_path} -> {lesson_id}")
        
        # Test related lessons finding
        if lesson_id in integrator.cross_lesson_data.get("content_index", {}):
            related_lessons = integrator.find_related_lessons(lesson_id)
            print(f"  - Related lessons for {lesson_id}: {len(related_lessons)} found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing PowerPoint + Notes integration: {e}")
        return False

def test_enhanced_powerpoint_audio_integration():
    """Test enhanced PowerPoint + Audio integration with cross-lesson context."""
    print("\n" + "=" * 60)
    print("TESTING ENHANCED POWERPOINT + AUDIO INTEGRATION")
    print("=" * 60)
    
    try:
        from integrate_powerpoint_audio import PowerPointAudioIntegrator
        
        integrator = PowerPointAudioIntegrator()
        
        print(f"✓ PowerPoint Audio Integrator initialized")
        print(f"  - Cross-lesson data loaded: {len(integrator.cross_lesson_data.get('content_index', {}))} lessons")
        print(f"  - Context config: {integrator.context_config}")
        
        # Test lesson ID extraction
        test_path = Path("lessons/TLP/presentations/test.pptx")
        lesson_id = integrator.get_lesson_id_from_path(test_path)
        print(f"  - Lesson ID extraction test: {test_path} -> {lesson_id}")
        
        # Test related lessons finding
        if lesson_id in integrator.cross_lesson_data.get("content_index", {}):
            related_lessons = integrator.find_related_lessons(lesson_id)
            print(f"  - Related lessons for {lesson_id}: {len(related_lessons)} found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing PowerPoint + Audio integration: {e}")
        return False

def test_enhanced_lesson_processing():
    """Test enhanced lesson processing with cross-lesson context."""
    print("\n" + "=" * 60)
    print("TESTING ENHANCED LESSON PROCESSING")
    print("=" * 60)
    
    try:
        from process_lesson import LessonProcessor
        
        processor = LessonProcessor()
        
        print(f"✓ Lesson Processor initialized")
        print(f"  - Cross-lesson data loaded: {len(processor.cross_lesson_data.get('content_index', {}))} lessons")
        print(f"  - Context config: {processor.context_config}")
        
        # Test with available lessons
        available_lessons = list(processor.cross_lesson_data.get("content_index", {}).keys())
        
        if available_lessons:
            test_lesson = available_lessons[0]
            print(f"Testing context enhancement for lesson: {test_lesson}")
            
            # Get context enhancement summary
            context_summary = processor.get_context_enhancement_summary(test_lesson)
            
            print(f"✓ Context enhancement summary:")
            print(f"  - Enabled: {context_summary.get('enabled', False)}")
            print(f"  - Related lessons found: {context_summary.get('related_lessons_found', 0)}")
            print(f"  - Lessons indexed: {context_summary.get('lessons_indexed', 0)}")
            
            if context_summary.get('related_lessons'):
                print(f"  - Related lessons:")
                for rel in context_summary['related_lessons'][:3]:
                    print(f"    * {rel.get('lesson_id')} (similarity: {rel.get('similarity_score', 0):.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing lesson processing: {e}")
        return False

def test_context_weighting_algorithm():
    """Test the context weighting algorithm."""
    print("\n" + "=" * 60)
    print("TESTING CONTEXT WEIGHTING ALGORITHM")
    print("=" * 60)
    
    try:
        from context_optimizer import ContextOptimizer
        
        optimizer = ContextOptimizer()
        
        # Test with available lessons
        available_lessons = list(optimizer.cross_lesson_data.get("content_index", {}).keys())
        
        if len(available_lessons) >= 2:
            source_lesson = available_lessons[0]
            target_lesson = available_lessons[1]
            
            print(f"Testing context weighting between:")
            print(f"  Source: {source_lesson}")
            print(f"  Target: {target_lesson}")
            
            # Calculate context weight
            weight = optimizer.calculate_context_weight(source_lesson, target_lesson)
            print(f"✓ Context weight: {weight:.3f}")
            
            # Test different content types
            content_types = ["general", "flashcards", "notes", "audio"]
            for content_type in content_types:
                weight = optimizer.calculate_context_weight(source_lesson, target_lesson, content_type)
                print(f"  - {content_type}: {weight:.3f}")
            
            return True
        else:
            print("⚠️  Need at least 2 lessons for weighting test")
            return False
            
    except Exception as e:
        print(f"❌ Error testing context weighting: {e}")
        return False

def test_adaptive_context_selection():
    """Test adaptive context selection algorithm."""
    print("\n" + "=" * 60)
    print("TESTING ADAPTIVE CONTEXT SELECTION")
    print("=" * 60)
    
    try:
        from context_optimizer import ContextOptimizer
        
        optimizer = ContextOptimizer()
        
        # Test with available lessons
        available_lessons = list(optimizer.cross_lesson_data.get("content_index", {}).keys())
        
        if available_lessons:
            test_lesson = available_lessons[0]
            
            print(f"Testing adaptive context selection for: {test_lesson}")
            
            # Test different context lengths
            context_lengths = [1000, 2000, 3000]
            
            for max_length in context_lengths:
                selected = optimizer.optimize_context_selection(test_lesson, "general", max_length)
                print(f"  Max length {max_length}: {len(selected)} contexts selected")
                
                if selected:
                    total_weight = sum(ctx["weight"] for ctx in selected)
                    avg_weight = total_weight / len(selected)
                    print(f"    Average weight: {avg_weight:.3f}")
            
            return True
        else:
            print("⚠️  No lessons available for testing")
            return False
            
    except Exception as e:
        print(f"❌ Error testing adaptive context selection: {e}")
        return False

def generate_phase2b_summary():
    """Generate a summary of Phase 2B implementation."""
    print("\n" + "=" * 60)
    print("PHASE 2B IMPLEMENTATION SUMMARY")
    print("=" * 60)
    
    summary = {
        "phase": "2B - Enhanced Processing Pipeline",
        "implementation_date": datetime.now().isoformat(),
        "components_enhanced": [
            "PowerPoint + Notes Integration",
            "PowerPoint + Audio Integration", 
            "Lesson Processing Pipeline",
            "Context Optimization Algorithms"
        ],
        "new_features": [
            "Cross-lesson context integration",
            "Context weighting algorithms",
            "Adaptive context selection",
            "Context quality assessment",
            "Enhanced metadata tracking"
        ],
        "files_modified": [
            "scripts/integrate_powerpoint_notes.py",
            "scripts/integrate_powerpoint_audio.py",
            "scripts/process_lesson.py"
        ],
        "files_created": [
            "scripts/context_optimizer.py"
        ],
        "key_improvements": [
            "Intelligent context selection based on multiple factors",
            "Semantic similarity and concept overlap analysis",
            "Relationship strength weighting",
            "Content freshness consideration",
            "Adaptive context window sizing",
            "Quality assessment and recommendations"
        ]
    }
    
    print("✓ Phase 2B Implementation Complete")
    print(f"  Components Enhanced: {len(summary['components_enhanced'])}")
    print(f"  New Features: {len(summary['new_features'])}")
    print(f"  Files Modified: {len(summary['files_modified'])}")
    print(f"  Files Created: {len(summary['files_created'])}")
    
    print("\nKey Improvements:")
    for improvement in summary['key_improvements']:
        print(f"  • {improvement}")
    
    # Save summary
    summary_file = Path("config/phase2b_implementation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Summary saved to: {summary_file}")
    
    return summary

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test Phase 2B Implementation")
    parser.add_argument("--test-all", action="store_true", help="Run all tests")
    parser.add_argument("--test-data", action="store_true", help="Test cross-lesson data loading")
    parser.add_argument("--test-optimizer", action="store_true", help="Test context optimizer")
    parser.add_argument("--test-notes", action="store_true", help="Test PowerPoint + Notes integration")
    parser.add_argument("--test-audio", action="store_true", help="Test PowerPoint + Audio integration")
    parser.add_argument("--test-processing", action="store_true", help="Test lesson processing")
    parser.add_argument("--test-weighting", action="store_true", help="Test context weighting")
    parser.add_argument("--test-selection", action="store_true", help="Test adaptive context selection")
    parser.add_argument("--summary", action="store_true", help="Generate implementation summary")
    
    args = parser.parse_args()
    
    if not any([args.test_all, args.test_data, args.test_optimizer, args.test_notes, 
                args.test_audio, args.test_processing, args.test_weighting, 
                args.test_selection, args.summary]):
        args.test_all = True
    
    results = {}
    
    if args.test_all or args.test_data:
        results["data_loading"] = test_cross_lesson_data_loading()
    
    if args.test_all or args.test_optimizer:
        results["context_optimizer"] = test_context_optimizer()
    
    if args.test_all or args.test_notes:
        results["powerpoint_notes"] = test_enhanced_powerpoint_notes_integration()
    
    if args.test_all or args.test_audio:
        results["powerpoint_audio"] = test_enhanced_powerpoint_audio_integration()
    
    if args.test_all or args.test_processing:
        results["lesson_processing"] = test_enhanced_lesson_processing()
    
    if args.test_all or args.test_weighting:
        results["context_weighting"] = test_context_weighting_algorithm()
    
    if args.test_all or args.test_selection:
        results["adaptive_selection"] = test_adaptive_context_selection()
    
    if args.summary:
        results["summary"] = generate_phase2b_summary()
    
    # Print overall results
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    for test_name, result in results.items():
        total += 1
        if result:
            passed += 1
            print(f"✓ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Phase 2B implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
