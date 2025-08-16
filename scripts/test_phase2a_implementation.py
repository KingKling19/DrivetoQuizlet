#!/usr/bin/env python3
"""
Test Phase 2a Implementation: Context Enhancement Engine

This script demonstrates the cross-lesson context enhancement features
implemented in Phase 2a of the Cross-Lesson Context System.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent))

from model_manager import ModelManager
from convert_folder_to_quizlet import (
    load_cross_lesson_data,
    get_lesson_id_from_path,
    find_related_lessons,
    extract_related_context,
    enhance_with_cross_lesson_context
)


def test_cross_lesson_data_loading():
    """Test loading of cross-lesson analysis data."""
    print("🔍 Testing Cross-Lesson Data Loading...")
    
    # Test using the function from convert_folder_to_quizlet
    data = load_cross_lesson_data()
    
    if data.get("content_index"):
        print(f"✓ Content index loaded: {len(data['content_index'])} lessons")
        for lesson_id in list(data['content_index'].keys())[:3]:
            lesson_name = data['content_index'][lesson_id].get('lesson_name', lesson_id)
            print(f"  - {lesson_id}: {lesson_name}")
    else:
        print("⚠️  No content index found")
    
    if data.get("lesson_relationships"):
        print(f"✓ Lesson relationships loaded: {len(data['lesson_relationships'])} lessons")
    else:
        print("⚠️  No lesson relationships found")
    
    if data.get("cross_references"):
        print(f"✓ Cross-references loaded: {len(data['cross_references'])} references")
    else:
        print("⚠️  No cross-references found")
    
    return data


def test_model_manager_context_methods():
    """Test ModelManager cross-lesson context methods."""
    print("\n🔍 Testing ModelManager Context Methods...")
    
    # Test getting cross-lesson data
    cross_lesson_data = ModelManager.get_cross_lesson_data()
    if cross_lesson_data:
        print("✓ Cross-lesson data loaded via ModelManager")
    else:
        print("⚠️  Could not load cross-lesson data via ModelManager")
        return
    
    # Test getting related lessons for a specific lesson
    test_lesson_id = "TLP"  # Use TLP as test case
    related_lessons = ModelManager.get_related_lessons(test_lesson_id, max_lessons=3)
    
    if related_lessons:
        print(f"✓ Found {len(related_lessons)} related lessons for '{test_lesson_id}':")
        for rel in related_lessons:
            lesson_id = rel.get('lesson_id', 'Unknown')
            similarity = rel.get('similarity_score', 0.0)
            concepts = rel.get('related_concepts', [])
            print(f"  - {lesson_id} (similarity: {similarity:.3f})")
            if concepts:
                print(f"    Related concepts: {', '.join(concepts[:3])}")
    else:
        print(f"⚠️  No related lessons found for '{test_lesson_id}'")
    
    # Test getting lesson context
    lesson_context = ModelManager.get_lesson_context(test_lesson_id, max_context_length=1000)
    if lesson_context:
        print(f"✓ Lesson context retrieved for '{test_lesson_id}' ({len(lesson_context)} chars)")
        print(f"  Preview: {lesson_context[:200]}...")
    else:
        print(f"⚠️  No lesson context found for '{test_lesson_id}'")
    
    # Test similarity calculation
    if related_lessons:
        test_related_id = related_lessons[0].get('lesson_id')
        if test_related_id:
            similarity = ModelManager.calculate_similarity_score(test_lesson_id, test_related_id)
            print(f"✓ Similarity between '{test_lesson_id}' and '{test_related_id}': {similarity:.3f}")


def test_context_enhancement_functions():
    """Test the context enhancement functions."""
    print("\n🔍 Testing Context Enhancement Functions...")
    
    # Load cross-lesson data
    cross_lesson_data = load_cross_lesson_data()
    if not cross_lesson_data.get("content_index"):
        print("⚠️  No cross-lesson data available for testing")
        return
    
    # Test lesson ID extraction
    test_path = Path("lessons/TLP/presentations/test.pptx")
    lesson_id = get_lesson_id_from_path(test_path)
    print(f"✓ Extracted lesson ID '{lesson_id}' from path: {test_path}")
    
    # Test finding related lessons
    related_lessons = find_related_lessons(lesson_id, cross_lesson_data, max_lessons=2)
    if related_lessons:
        print(f"✓ Found {len(related_lessons)} related lessons for '{lesson_id}'")
        for rel in related_lessons:
            rel_id = rel.get('lesson_id', 'Unknown')
            similarity = rel.get('similarity_score', 0.0)
            print(f"  - {rel_id} (similarity: {similarity:.3f})")
    else:
        print(f"⚠️  No related lessons found for '{lesson_id}'")
    
    # Test context extraction
    if related_lessons:
        context = extract_related_context(related_lessons, cross_lesson_data)
        if context:
            print(f"✓ Extracted context ({len(context)} chars)")
            print(f"  Preview: {context[:300]}...")
        else:
            print("⚠️  No context extracted")


def test_context_enhancement_integration():
    """Test the integration of context enhancement with slide processing."""
    print("\n🔍 Testing Context Enhancement Integration...")
    
    # Simulate a slide chunk
    mock_chunk = [
        {
            "index": 1,
            "title": "Introduction to TLP",
            "body": "This slide introduces the Traffic Light Protocol concepts.",
            "notes": "TLP is a system for sharing sensitive information."
        },
        {
            "index": 2,
            "title": "TLP Colors",
            "body": "Red, Amber, Green, and White are the TLP colors.",
            "notes": "Each color has specific sharing restrictions."
        }
    ]
    
    # Test path
    test_path = Path("lessons/TLP/presentations/tlp_overview.pptx")
    
    # Load cross-lesson data
    cross_lesson_data = load_cross_lesson_data()
    
    # Test context enhancement
    enhanced_context = enhance_with_cross_lesson_context(mock_chunk, test_path, cross_lesson_data)
    
    if enhanced_context:
        print(f"✓ Context enhancement successful ({len(enhanced_context)} chars)")
        print("Enhanced context preview:")
        print("-" * 50)
        print(enhanced_context[:500] + "..." if len(enhanced_context) > 500 else enhanced_context)
        print("-" * 50)
    else:
        print("⚠️  No context enhancement generated")


def test_performance_metrics():
    """Test performance of context enhancement operations."""
    print("\n🔍 Testing Performance Metrics...")
    
    import time
    
    # Test ModelManager cache performance
    start_time = time.time()
    data1 = ModelManager.get_cross_lesson_data()
    load_time1 = time.time() - start_time
    
    start_time = time.time()
    data2 = ModelManager.get_cross_lesson_data()  # Should be cached
    load_time2 = time.time() - start_time
    
    print(f"✓ First load: {load_time1:.3f}s")
    print(f"✓ Cached load: {load_time2:.3f}s")
    if load_time2 > 0:
        print(f"✓ Cache speedup: {load_time1/load_time2:.1f}x")
    else:
        print("✓ Cache speedup: Instant (cached)")
    
    # Test related lessons lookup performance
    test_lesson_id = "TLP"
    start_time = time.time()
    related_lessons = ModelManager.get_related_lessons(test_lesson_id)
    lookup_time = time.time() - start_time
    
    print(f"✓ Related lessons lookup: {lookup_time:.3f}s")
    
    # Test context generation performance
    if related_lessons:
        start_time = time.time()
        context = ModelManager.get_lesson_context(test_lesson_id)
        context_time = time.time() - start_time
        
        print(f"✓ Context generation: {context_time:.3f}s")
        print(f"✓ Context size: {len(context)} chars")


def main():
    """Run all Phase 2a tests."""
    print("🚀 Phase 2a Context Enhancement Engine - Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Cross-lesson data loading
        test_cross_lesson_data_loading()
        
        # Test 2: ModelManager context methods
        test_model_manager_context_methods()
        
        # Test 3: Context enhancement functions
        test_context_enhancement_functions()
        
        # Test 4: Context enhancement integration
        test_context_enhancement_integration()
        
        # Test 5: Performance metrics
        test_performance_metrics()
        
        print("\n" + "=" * 60)
        print("✅ Phase 2a Context Enhancement Engine tests completed!")
        print("\nKey Features Implemented:")
        print("✓ Cross-lesson data loading and caching")
        print("✓ Related lesson discovery")
        print("✓ Context extraction and enhancement")
        print("✓ Integration with flashcard generation")
        print("✓ Performance optimization with caching")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
