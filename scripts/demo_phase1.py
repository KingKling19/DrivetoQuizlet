#!/usr/bin/env python3
"""
Phase 1 Demonstration Script

Demonstrates the capabilities of the Cross-Lesson Context System Phase 1 implementation.
Shows content indexing, similarity analysis, and relationship detection in action.
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from lesson_content_indexer import LessonContentIndexer
    from cross_lesson_analyzer import CrossLessonAnalyzer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure lesson_content_indexer.py and cross_lesson_analyzer.py are in the scripts directory")
    sys.exit(1)


def demo_content_indexing():
    """Demonstrate content indexing capabilities."""
    print("🔍 Content Indexing Demonstration")
    print("=" * 50)
    
    indexer = LessonContentIndexer()
    
    # Show indexed lessons
    lesson_ids = indexer.get_all_lesson_ids()
    print(f"📚 Indexed Lessons: {len(lesson_ids)}")
    for lesson_id in lesson_ids:
        content = indexer.get_lesson_content(lesson_id)
        if content:
            print(f"  - {lesson_id}")
            print(f"    Sources: {list(content['content_sources'].keys())}")
            print(f"    Concepts: {len(content.get('key_concepts', []))}")
            print(f"    Fingerprint: {len(content.get('content_fingerprint', []))} dimensions")
            print(f"    Embedding: {len(content.get('semantic_embedding', []))} dimensions")
    
    # Demonstrate search
    print(f"\n🔍 Content Search Examples:")
    search_terms = ["TLP", "operations", "degraded"]
    for term in search_terms:
        results = indexer.search_content(term, top_k=2)
        print(f"  Search for '{term}':")
        for lesson_id, score in results:
            print(f"    - {lesson_id}: score {score}")


def demo_similarity_analysis():
    """Demonstrate similarity analysis capabilities."""
    print(f"\n🔗 Similarity Analysis Demonstration")
    print("=" * 50)
    
    analyzer = CrossLessonAnalyzer()
    
    if not analyzer.content_index:
        print("❌ No content index found. Run content indexing first.")
        return
    
    lesson_ids = list(analyzer.content_index.keys())
    
    # Show similarity matrix
    print("📊 Similarity Matrix:")
    for i, lesson1 in enumerate(lesson_ids):
        print(f"  {lesson1}:")
        for lesson2 in lesson_ids:
            if lesson1 != lesson2:
                similarity = analyzer.calculate_content_similarity(lesson1, lesson2)
                print(f"    → {lesson2}: {similarity:.3f}")
    
    # Show cross-references
    print(f"\n🔗 Cross-References:")
    for lesson_id in lesson_ids:
        cross_refs = analyzer.detect_cross_references(lesson_id)
        if cross_refs:
            print(f"  {lesson_id}:")
            for ref in cross_refs:
                print(f"    → {ref['target_lesson']}: {len(ref['overlapping_concepts'])} concepts")


def demo_context_recommendations():
    """Demonstrate context recommendation capabilities."""
    print(f"\n🎯 Context Recommendations Demonstration")
    print("=" * 50)
    
    analyzer = CrossLessonAnalyzer()
    
    if not analyzer.lesson_relationships:
        print("❌ No relationship data found. Run analysis first.")
        return
    
    lesson_ids = list(analyzer.lesson_relationships.keys())
    
    for lesson_id in lesson_ids:
        print(f"\n📚 Context recommendations for {lesson_id}:")
        recommendations = analyzer.get_context_recommendations(lesson_id)
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec['lesson_id']}")
                print(f"     Weight: {rec['context_weight']:.3f}")
                print(f"     Type: {rec['relationship_type']}")
                print(f"     Context snippets: {len(rec['recommended_context'])}")
        else:
            print("  No strong context recommendations found")


def demo_relationship_analysis():
    """Demonstrate relationship analysis capabilities."""
    print(f"\n🔗 Relationship Analysis Demonstration")
    print("=" * 50)
    
    analyzer = CrossLessonAnalyzer()
    
    if not analyzer.lesson_relationships:
        print("❌ No relationship data found. Run analysis first.")
        return
    
    # Print analysis summary
    analyzer.print_analysis_summary()
    
    # Show detailed relationships
    print(f"\n📋 Detailed Relationship Analysis:")
    for lesson_id, relationships in analyzer.lesson_relationships.items():
        print(f"\n{lesson_id}:")
        
        if relationships["prerequisites"]:
            print(f"  Prerequisites: {', '.join(relationships['prerequisites'])}")
        
        if relationships["complementary_lessons"]:
            print(f"  Complementary: {', '.join(relationships['complementary_lessons'])}")
        
        if relationships["related_lessons"]:
            print(f"  Related: {', '.join(relationships['related_lessons'])}")
        
        # Show top similarity scores
        top_scores = sorted(
            relationships["relationship_scores"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        if top_scores:
            print(f"  Top similarities:")
            for other_id, score in top_scores:
                rel_type = analyzer._get_relationship_type(other_id, relationships)
                print(f"    → {other_id} ({rel_type}): {score:.3f}")


def main():
    """Run the Phase 1 demonstration."""
    print("🚀 Cross-Lesson Context System - Phase 1 Demonstration")
    print("=" * 70)
    print("This demonstration showcases the capabilities of the Phase 1 implementation")
    print("which provides content indexing, similarity analysis, and relationship detection.")
    print()
    
    try:
        # Run demonstrations
        demo_content_indexing()
        demo_similarity_analysis()
        demo_context_recommendations()
        demo_relationship_analysis()
        
        print(f"\n{'='*70}")
        print("✅ Phase 1 Demonstration Complete!")
        print("\nKey Capabilities Demonstrated:")
        print("  📚 Content Indexing: Multi-source content extraction and fingerprinting")
        print("  🔍 Similarity Analysis: Multi-method similarity calculation")
        print("  🔗 Cross-References: Concept overlap and detailed reference detection")
        print("  🎯 Context Recommendations: Weighted context enhancement suggestions")
        print("  📊 Relationship Analysis: Comprehensive lesson relationship mapping")
        print("\nReady for Phase 2: Context Enhancement Engine")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        print("Make sure Phase 1 implementation is complete and all dependencies are installed.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
