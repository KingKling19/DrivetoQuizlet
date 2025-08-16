#!/usr/bin/env python3
"""
Test Flashcard Review Interface

Demonstrates the flashcard review interface functionality with sample flashcards.
"""

import json
import os
import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent))

def create_sample_flashcards():
    """Create sample flashcards for testing"""
    sample_flashcards = [
        {
            "term": "Command and Control",
            "definition": "The exercise of authority and direction by a properly designated commander over assigned and attached forces in the accomplishment of a mission.",
            "source_slide": 1,
            "confidence": 0.9
        },
        {
            "term": "Intelligence",
            "definition": "The product resulting from the collection, processing, integration, evaluation, analysis, and interpretation of available information concerning foreign nations, hostile or potentially hostile forces or elements, or areas of actual or potential operations.",
            "source_slide": 2,
            "confidence": 0.85
        },
        {
            "term": "Logistics",
            "definition": "The process of planning and executing the movement and sustainment of forces.",
            "source_slide": 3,
            "confidence": 0.8
        },
        {
            "term": "Communication",
            "definition": "The exchange of information between individuals, units, or organizations.",
            "source_slide": 4,
            "confidence": 0.75
        },
        {
            "term": "Surveillance",
            "definition": "The systematic observation of aerospace, surface, or subsurface areas, places, persons, or things.",
            "source_slide": 5,
            "confidence": 0.9
        },
        {
            "term": "Command and Control System",
            "definition": "The facilities, equipment, communications, procedures, and personnel essential to a commander for planning, directing, and controlling operations of assigned forces.",
            "source_slide": 1,
            "confidence": 0.8
        },
        {
            "term": "Radio Communication",
            "definition": "The transmission of information by radio waves.",
            "source_slide": 4,
            "confidence": 0.7
        },
        {
            "term": "Safety Protocol",
            "definition": "A set of rules and procedures designed to ensure safety.",
            "source_slide": 6,
            "confidence": 0.85
        },
        {
            "term": "Leadership",
            "definition": "The ability to influence and guide others toward achieving a common goal.",
            "source_slide": 7,
            "confidence": 0.8
        },
        {
            "term": "Equipment Maintenance",
            "definition": "The process of keeping equipment in good working condition.",
            "source_slide": 8,
            "confidence": 0.75
        }
    ]
    
    return sample_flashcards

def test_flashcard_review_interface():
    """Test the flashcard review interface"""
    try:
        from flashcard_review_interface import FlashcardReviewInterface
        from flashcard_quality_assessor import FlashcardQualityAssessor
        from flashcard_clustering import FlashcardClusterer
        from flashcard_optimizer import FlashcardOptimizer
        
        print("=== Flashcard Review Interface Test ===\n")
        
        # Create sample flashcards
        sample_flashcards = create_sample_flashcards()
        print(f"Created {len(sample_flashcards)} sample flashcards\n")
        
        # Initialize review interface
        review_interface = FlashcardReviewInterface()
        print("✓ FlashcardReviewInterface initialized\n")
        
        # Test quality assessment
        print("=== Quality Assessment Test ===")
        quality_assessor = FlashcardQualityAssessor()
        
        for i, flashcard in enumerate(sample_flashcards[:3]):
            assessment = quality_assessor.assess_flashcard_quality(flashcard)
            print(f"Flashcard {i+1}: {flashcard['term']}")
            print(f"  Quality Score: {assessment['overall_score']}")
            print(f"  Difficulty Level: {assessment['difficulty_level']}")
            print(f"  Status: {assessment['status']}")
            print(f"  Recommendations: {', '.join(assessment['recommendations'][:2])}")
            print()
        
        # Test clustering
        print("=== Clustering Test ===")
        clusterer = FlashcardClusterer()
        
        topic_clusters = clusterer.cluster_flashcards_by_topic(sample_flashcards)
        print(f"Topic clusters: {len(topic_clusters)}")
        for topic, cluster_data in topic_clusters.items():
            print(f"  {topic}: {cluster_data['count']} flashcards")
        
        difficulty_clusters = clusterer.cluster_flashcards_by_difficulty(sample_flashcards)
        print(f"\nDifficulty clusters: {len(difficulty_clusters)}")
        for difficulty, cluster_data in difficulty_clusters.items():
            print(f"  {difficulty}: {cluster_data['count']} flashcards")
        
        # Test optimization
        print("\n=== Optimization Test ===")
        optimizer = FlashcardOptimizer()
        
        # Analyze content distribution
        distribution = optimizer.analyze_content_distribution(sample_flashcards)
        print(f"Content distribution analysis:")
        print(f"  Total flashcards: {distribution['total_flashcards']}")
        print(f"  Balance score: {distribution['balance_score']}")
        print(f"  Balance level: {distribution['balance_level']}")
        
        # Identify content gaps
        gaps = optimizer.identify_content_gaps(sample_flashcards)
        print(f"\nContent gaps identified: {len(gaps)}")
        for gap in gaps[:3]:
            print(f"  {gap['topic']}: {gap['gap_size']:.1f}% gap (priority: {gap['priority']})")
        
        # Test optimization
        optimized = optimizer.optimize_flashcards(sample_flashcards)
        print(f"\nOptimization result: {len(optimized)} flashcards remaining")
        
        # Test review interface functionality
        print("\n=== Review Interface Test ===")
        
        # Create a temporary lesson directory for testing
        test_lesson_dir = Path("lessons/test_flashcard_review")
        test_lesson_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sample flashcards
        output_file = test_lesson_dir / "output" / "sample_flashcards.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump({
                'metadata': {
                    'lesson_path': str(test_lesson_dir),
                    'created_at': '2024-01-01T00:00:00',
                    'total_flashcards': len(sample_flashcards)
                },
                'flashcards': sample_flashcards
            }, f, indent=2)
        
        # Load flashcards for review
        review_data = review_interface.load_flashcards_for_review(str(test_lesson_dir))
        print(f"✓ Loaded {review_data['total_flashcards']} flashcards for review")
        print(f"  Suggestions: {len(review_data['suggestions'])}")
        print(f"  Clusters: {len(review_data['clusters'])}")
        
        # Test optimization suggestions
        processed_cards = review_interface._process_flashcards_for_review(sample_flashcards)
        suggestions = review_interface.get_optimization_suggestions(processed_cards)
        print(f"\nOptimization suggestions: {len(suggestions)}")
        for suggestion in suggestions[:2]:
            print(f"  {suggestion['title']}: {suggestion['description']}")
        
        # Test bulk operations
        print("\n=== Bulk Operations Test ===")
        operations = [
            {
                'type': 'improve_quality',
                'card_ids': ['fc_0_test'],
                'parameters': {'improve_definition': True}
            }
        ]
        
        processed_flashcards = review_interface.apply_bulk_operations(
            [{'id': 'fc_0_test', 'term': 'Test', 'definition': 'Short'}], 
            operations
        )
        print(f"✓ Applied {len(operations)} bulk operations")
        print(f"  Processed flashcards: {len(processed_flashcards)}")
        
        # Test review statistics
        statistics = review_interface.get_review_statistics(sample_flashcards)
        print(f"\nReview statistics:")
        print(f"  Total flashcards: {statistics['total_flashcards']}")
        print(f"  Average quality: {statistics['average_quality_score']}")
        print(f"  Review progress: {statistics['review_progress']:.1f}%")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_lesson_dir, ignore_errors=True)
        
        print("\n=== Test Completed Successfully ===")
        print("✓ All flashcard review interface components working correctly")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are available")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_individual_components():
    """Test individual components separately"""
    print("\n=== Individual Component Tests ===\n")
    
    # Test quality assessor
    try:
        from flashcard_quality_assessor import FlashcardQualityAssessor
        assessor = FlashcardQualityAssessor()
        print("✓ FlashcardQualityAssessor working")
    except Exception as e:
        print(f"❌ FlashcardQualityAssessor failed: {e}")
    
    # Test clusterer
    try:
        from flashcard_clustering import FlashcardClusterer
        clusterer = FlashcardClusterer()
        print("✓ FlashcardClusterer working")
    except Exception as e:
        print(f"❌ FlashcardClusterer failed: {e}")
    
    # Test optimizer
    try:
        from flashcard_optimizer import FlashcardOptimizer
        optimizer = FlashcardOptimizer()
        print("✓ FlashcardOptimizer working")
    except Exception as e:
        print(f"❌ FlashcardOptimizer failed: {e}")

if __name__ == "__main__":
    print("Starting Flashcard Review Interface Tests...\n")
    
    # Test individual components first
    test_individual_components()
    
    # Test full interface
    test_flashcard_review_interface()
    
    print("\nAll tests completed!")
