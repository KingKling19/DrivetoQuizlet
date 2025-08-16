#!/usr/bin/env python3
"""
Test Phase 1 Implementation

Validates the Phase 1 implementation of the Cross-Lesson Context System.
Tests content indexing, similarity analysis, and relationship detection.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from lesson_content_indexer import LessonContentIndexer
    from cross_lesson_analyzer import CrossLessonAnalyzer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure lesson_content_indexer.py and cross_lesson_analyzer.py are in the scripts directory")
    sys.exit(1)


def test_content_indexing():
    """Test the content indexing functionality."""
    print("🧪 Testing Content Indexing...")
    
    try:
        # Initialize indexer
        indexer = LessonContentIndexer()
        
        # Check if lessons directory exists
        if not indexer.lessons_dir.exists():
            print(f"⚠️  Lessons directory not found: {indexer.lessons_dir}")
            return False
        
        # List available lessons
        lesson_dirs = [d for d in indexer.lessons_dir.iterdir() if d.is_dir()]
        print(f"📚 Found {len(lesson_dirs)} lesson directories: {[d.name for d in lesson_dirs]}")
        
        if not lesson_dirs:
            print("❌ No lesson directories found")
            return False
        
        # Test indexing a single lesson
        test_lesson = lesson_dirs[0]
        print(f"🔍 Testing indexing for: {test_lesson.name}")
        
        content = indexer.extract_lesson_content(test_lesson)
        
        # Validate content structure
        required_fields = ["lesson_id", "lesson_name", "content_sources", "key_concepts"]
        for field in required_fields:
            if field not in content:
                print(f"❌ Missing required field: {field}")
                return False
        
        print(f"✓ Content extraction successful")
        print(f"  - Lesson ID: {content['lesson_id']}")
        print(f"  - Content sources: {list(content['content_sources'].keys())}")
        print(f"  - Key concepts: {len(content['key_concepts'])}")
        
        # Test fingerprint generation
        fingerprint = indexer.generate_content_fingerprint(content)
        if not fingerprint or len(fingerprint) == 0:
            print("❌ Fingerprint generation failed")
            return False
        
        print(f"✓ Fingerprint generation successful (length: {len(fingerprint)})")
        
        # Test semantic embedding generation
        embedding = indexer.generate_semantic_embedding(content)
        if not embedding or len(embedding) == 0:
            print("⚠️  Semantic embedding generation failed (may be due to API issues)")
        else:
            print(f"✓ Semantic embedding generation successful (length: {len(embedding)})")
        
        return True
        
    except Exception as e:
        print(f"❌ Content indexing test failed: {e}")
        return False


def test_cross_lesson_analysis():
    """Test the cross-lesson analysis functionality."""
    print("\n🧪 Testing Cross-Lesson Analysis...")
    
    try:
        # Initialize analyzer
        analyzer = CrossLessonAnalyzer()
        
        # Check if content index exists
        if not analyzer.content_index:
            print("⚠️  No content index found. Run content indexing first.")
            return False
        
        print(f"📊 Found {len(analyzer.content_index)} lessons in index")
        
        # Test similarity calculation between two lessons
        lesson_ids = list(analyzer.content_index.keys())
        if len(lesson_ids) < 2:
            print("⚠️  Need at least 2 lessons for similarity testing")
            return False
        
        lesson1_id = lesson_ids[0]
        lesson2_id = lesson_ids[1]
        
        print(f"🔍 Testing similarity between {lesson1_id} and {lesson2_id}")
        
        similarity = analyzer.calculate_content_similarity(lesson1_id, lesson2_id)
        print(f"✓ Similarity score: {similarity:.3f}")
        
        # Test cross-reference detection
        print(f"🔗 Testing cross-reference detection for {lesson1_id}")
        cross_refs = analyzer.detect_cross_references(lesson1_id)
        print(f"✓ Found {len(cross_refs)} cross-references")
        
        for ref in cross_refs[:2]:  # Show first 2
            print(f"  - {ref['target_lesson']}: {len(ref['overlapping_concepts'])} overlapping concepts")
        
        # Test context recommendations
        print(f"🎯 Testing context recommendations for {lesson1_id}")
        recommendations = analyzer.get_context_recommendations(lesson1_id)
        print(f"✓ Generated {len(recommendations)} context recommendations")
        
        for rec in recommendations[:2]:  # Show first 2
            print(f"  - {rec['lesson_id']} (weight: {rec['context_weight']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Cross-lesson analysis test failed: {e}")
        return False


def test_configuration_loading():
    """Test configuration file loading."""
    print("\n🧪 Testing Configuration Loading...")
    
    try:
        config_file = Path("config/lesson_relationships.json")
        
        if not config_file.exists():
            print(f"⚠️  Configuration file not found: {config_file}")
            return False
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Validate configuration structure
        required_sections = ["lesson_hierarchy", "content_correlation_weights", "context_window_parameters"]
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing configuration section: {section}")
                return False
        
        print(f"✓ Configuration loaded successfully")
        print(f"  - Lesson hierarchy: {len(config['lesson_hierarchy'])} lessons")
        print(f"  - Correlation weights: {len(config['content_correlation_weights'])} weights")
        print(f"  - Context parameters: {len(config['context_window_parameters'])} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        return False


def test_full_pipeline():
    """Test the full Phase 1 pipeline."""
    print("\n🧪 Testing Full Phase 1 Pipeline...")
    
    try:
        # Step 1: Content Indexing
        print("📚 Step 1: Content Indexing")
        indexer = LessonContentIndexer()
        
        # Check if index already exists
        if indexer.index_file.exists():
            print("✓ Content index already exists, loading...")
        else:
            print("🔍 Creating new content index...")
            indexer.index_all_lessons()
        
        # Step 2: Cross-Lesson Analysis
        print("\n🔗 Step 2: Cross-Lesson Analysis")
        analyzer = CrossLessonAnalyzer()
        
        # Check if analysis already exists
        if analyzer.relationships_file.exists():
            print("✓ Analysis results already exist, loading...")
        else:
            print("🔍 Performing cross-lesson analysis...")
            analyzer.analyze_all_lessons()
        
        # Step 3: Validate Results
        print("\n✅ Step 3: Validating Results")
        
        # Check files exist
        required_files = [
            "config/lesson_content_index.json",
            "config/semantic_embeddings.pkl",
            "config/lesson_relationships.json",
            "config/lesson_similarity_matrix.json",
            "config/cross_references.json"
        ]
        
        for file_path in required_files:
            if Path(file_path).exists():
                print(f"✓ {file_path}")
            else:
                print(f"❌ {file_path} - Missing")
                return False
        
        # Print summary
        analyzer.print_analysis_summary()
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False


def main():
    """Run all Phase 1 tests."""
    print("🚀 Phase 1 Implementation Test Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration Loading", test_configuration_loading),
        ("Content Indexing", test_content_indexing),
        ("Cross-Lesson Analysis", test_cross_lesson_analysis),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results[test_name] = success
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 1 tests passed! Ready for Phase 2.")
    else:
        print("⚠️  Some tests failed. Please review and fix issues before proceeding to Phase 2.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
