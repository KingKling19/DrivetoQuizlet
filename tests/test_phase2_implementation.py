"""
Test Phase 2 Implementation: Content Balance & Clustering System

This script tests the Phase 2 implementation of the flashcard optimization system,
including content balance analysis and clustering capabilities.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent))

from flashcard_optimizer import FlashcardOptimizer
from flashcard_clustering import FlashcardClustering
from flashcard_quality_assessor import FlashcardQualityAssessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_flashcards() -> List[Dict[str, Any]]:
    """Create sample flashcards for testing."""
    return [
        {
            'term': 'Command and Control',
            'definition': 'The exercise of authority and direction by a properly designated commander over assigned and attached forces.',
            'confidence': 0.9,
            'source': 'slide_1',
            'slide_number': 1
        },
        {
            'term': 'Intelligence',
            'definition': 'The product resulting from the collection, processing, integration, evaluation, analysis, and interpretation of available information.',
            'confidence': 0.8,
            'source': 'slide_2',
            'slide_number': 2
        },
        {
            'term': 'Logistics',
            'definition': 'The process of planning and executing the movement and sustainment of forces.',
            'confidence': 0.7,
            'source': 'slide_3',
            'slide_number': 3
        },
        {
            'term': 'Communication',
            'definition': 'The exchange of information between individuals, units, or organizations.',
            'confidence': 0.85,
            'source': 'notes_1',
            'slide_number': None
        },
        {
            'term': 'Surveillance',
            'definition': 'The systematic observation of aerospace, surface, or subsurface areas, places, persons, or things.',
            'confidence': 0.75,
            'source': 'notes_2',
            'slide_number': None
        },
        {
            'term': 'Advanced Tactical Procedures',
            'definition': 'Complex operational protocols requiring extensive training and specialized knowledge in military doctrine and strategic planning.',
            'confidence': 0.6,
            'source': 'slide_4',
            'slide_number': 4
        },
        {
            'term': 'Basic Communication Protocol',
            'definition': 'Standard procedures for transmitting information between units.',
            'confidence': 0.95,
            'source': 'slide_5',
            'slide_number': 5
        },
        {
            'term': 'Strategic Planning',
            'definition': 'The process of developing long-term objectives and determining the best approach to achieve them.',
            'confidence': 0.8,
            'source': 'slide_6',
            'slide_number': 6
        }
    ]


def create_sample_lesson_content() -> Dict[str, Any]:
    """Create sample lesson content for testing."""
    return {
        'title': 'Military Operations and Communications',
        'description': 'Comprehensive overview of military command, control, and communication systems',
        'summary': 'This lesson covers the fundamental principles of military operations including command and control, intelligence gathering, logistics management, and communication protocols.',
        'slides': [
            'Introduction to Command and Control',
            'Intelligence Systems and Procedures',
            'Logistics and Supply Chain Management',
            'Communication Protocols and Standards',
            'Advanced Tactical Procedures',
            'Strategic Planning and Execution'
        ],
        'notes': [
            'Communication is critical for successful operations',
            'Intelligence provides the foundation for decision making',
            'Logistics ensures operational readiness'
        ]
    }


def test_content_balance_analysis():
    """Test content balance analysis functionality."""
    logger.info("=== Testing Content Balance Analysis ===")
    
    optimizer = FlashcardOptimizer()
    flashcards = create_sample_flashcards()
    lesson_content = create_sample_lesson_content()
    
    # Test content distribution analysis
    print("\n1. Content Distribution Analysis:")
    distribution = optimizer.analyze_content_distribution(flashcards)
    
    print(f"   Total flashcards: {distribution['distribution_metrics']['total_flashcards']}")
    print(f"   Unique topics: {distribution['distribution_metrics']['unique_topics']}")
    print(f"   Balance score: {distribution['distribution_metrics']['balance_score']:.3f}")
    print(f"   Average coverage: {distribution['distribution_metrics']['average_coverage']:.3f}")
    
    print("\n   Topic Distribution:")
    for topic, info in distribution['topic_distribution'].items():
        print(f"   - {topic}: {info['coverage']:.3f} coverage ({info['status']})")
    
    # Test content gap identification
    print("\n2. Content Gap Identification:")
    gaps = optimizer.identify_content_gaps(flashcards, lesson_content)
    
    print(f"   Found {len(gaps)} content gaps:")
    for gap in gaps:
        print(f"   - {gap['topic']}: {gap['current_coverage']:.3f} coverage (priority: {gap['priority']})")
        if gap.get('missing_from_lesson'):
            print(f"     (Missing from lesson content)")
    
    # Test balance recommendations
    print("\n3. Balance Recommendations:")
    recommendations = optimizer.balance_content_coverage(flashcards)
    
    print(f"   Generated {len(recommendations)} recommendations:")
    for rec in recommendations:
        print(f"   - {rec['type']}: {rec['description']}")
        print(f"     Priority: {rec['priority']}")
        if rec['type'] == 'add_flashcards':
            print(f"     Estimated flashcards needed: {rec['estimated_flashcards_needed']}")
    
    # Test topic coverage assessment
    print("\n4. Topic Coverage Assessment:")
    coverage = optimizer.assess_topic_coverage(flashcards)
    
    print(f"   Coverage percentage: {coverage['coverage_percentage']:.3f}")
    print(f"   Overall score: {coverage['overall_score']:.3f}")
    print(f"   Covered topics: {len(coverage['covered_topics'])}")
    print(f"   Missing topics: {len(coverage['missing_topics'])}")
    
    return {
        'distribution': distribution,
        'gaps': gaps,
        'recommendations': recommendations,
        'coverage': coverage
    }


def test_clustering_system():
    """Test clustering system functionality."""
    logger.info("=== Testing Clustering System ===")
    
    clustering = FlashcardClustering()
    flashcards = create_sample_flashcards()
    
    # Test topic clustering
    print("\n1. Topic Clustering:")
    topic_clusters = clustering.cluster_flashcards_by_topic(flashcards)
    
    print(f"   Created {topic_clusters['n_clusters']} topic clusters:")
    for cluster_name, cluster_flashcards in topic_clusters['clusters'].items():
        print(f"   - {cluster_name}: {len(cluster_flashcards)} flashcards")
        if cluster_name in topic_clusters['summaries']:
            summary = topic_clusters['summaries'][cluster_name]
            print(f"     Top terms: {', '.join(summary['top_terms'])}")
            print(f"     Average confidence: {summary['average_confidence']:.3f}")
    
    # Test difficulty clustering
    print("\n2. Difficulty Clustering:")
    difficulty_clusters = clustering.cluster_flashcards_by_difficulty(flashcards)
    
    print("   Difficulty distribution:")
    for difficulty, cluster_flashcards in difficulty_clusters['clusters'].items():
        print(f"   - {difficulty}: {len(cluster_flashcards)} flashcards")
        if difficulty in difficulty_clusters['summaries']:
            summary = difficulty_clusters['summaries'][difficulty]
            print(f"     Average difficulty: {summary['average_difficulty']:.3f}")
            print(f"     Difficulty range: {summary['difficulty_range']}")
    
    # Test source clustering
    print("\n3. Source Clustering:")
    source_clusters = clustering.cluster_flashcards_by_source(flashcards)
    
    print("   Source distribution:")
    for source, cluster_flashcards in source_clusters['clusters'].items():
        print(f"   - {source}: {len(cluster_flashcards)} flashcards")
        if source in source_clusters['summaries']:
            summary = source_clusters['summaries'][source]
            print(f"     Average confidence: {summary['average_confidence']:.3f}")
            print(f"     Unique terms: {summary['unique_terms']}")
    
    # Test cluster summaries
    print("\n4. Cluster Summaries:")
    all_clusters = {
        **topic_clusters['clusters'],
        **difficulty_clusters['clusters'],
        **source_clusters['clusters']
    }
    summaries = clustering.generate_cluster_summaries(all_clusters)
    
    print(f"   Generated summaries for {summaries['overall_statistics']['total_clusters']} clusters")
    print(f"   Total flashcards analyzed: {summaries['overall_statistics']['total_flashcards']}")
    
    return {
        'topic_clusters': topic_clusters,
        'difficulty_clusters': difficulty_clusters,
        'source_clusters': source_clusters,
        'summaries': summaries
    }


def test_integration_with_quality_assessor():
    """Test integration with the quality assessor from Phase 1."""
    logger.info("=== Testing Integration with Quality Assessor ===")
    
    quality_assessor = FlashcardQualityAssessor()
    optimizer = FlashcardOptimizer()
    clustering = FlashcardClustering()
    
    flashcards = create_sample_flashcards()
    
    # Assess quality of all flashcards
    print("\n1. Quality Assessment:")
    quality_results = []
    for flashcard in flashcards:
        quality = quality_assessor.assess_flashcard_quality(flashcard)
        quality_results.append(quality)
        print(f"   {flashcard['term']}: Quality score = {quality['overall_score']:.3f}")
    
    # Analyze content distribution
    print("\n2. Content Balance Analysis:")
    distribution = optimizer.analyze_content_distribution(flashcards)
    print(f"   Balance score: {distribution['distribution_metrics']['balance_score']:.3f}")
    
    # Cluster by topic and analyze quality within clusters
    print("\n3. Quality Analysis by Topic Clusters:")
    topic_clusters = clustering.cluster_flashcards_by_topic(flashcards)
    
    for cluster_name, cluster_flashcards in topic_clusters['clusters'].items():
        cluster_quality_scores = []
        for flashcard in cluster_flashcards:
            # Find corresponding quality result
            for i, original_flashcard in enumerate(flashcards):
                if original_flashcard['term'] == flashcard['term']:
                    cluster_quality_scores.append(quality_results[i]['overall_score'])
                    break
        
        if cluster_quality_scores:
            avg_quality = sum(cluster_quality_scores) / len(cluster_quality_scores)
            print(f"   {cluster_name}: Average quality = {avg_quality:.3f} ({len(cluster_flashcards)} flashcards)")
    
    return {
        'quality_results': quality_results,
        'distribution': distribution,
        'topic_clusters': topic_clusters
    }


def test_configuration_loading():
    """Test configuration loading and validation."""
    logger.info("=== Testing Configuration Loading ===")
    
    try:
        # Test optimizer configuration
        optimizer = FlashcardOptimizer()
        print(f"   Optimizer config loaded successfully")
        print(f"   Content balance settings: {optimizer.config['content_balance']}")
        
        # Test clustering configuration
        clustering = FlashcardClustering()
        print(f"   Clustering config loaded successfully")
        print(f"   Clustering settings: {clustering.config['clustering']}")
        
        # Test quality assessor configuration
        quality_assessor = FlashcardQualityAssessor()
        print(f"   Quality assessor config loaded successfully")
        print(f"   Quality scoring weights: {quality_assessor.config['quality_scoring']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration loading failed: {e}")
        return False


def main():
    """Run all Phase 2 tests."""
    logger.info("Starting Phase 2 Implementation Tests")
    
    # Test configuration loading
    config_success = test_configuration_loading()
    if not config_success:
        logger.error("Configuration loading failed. Exiting.")
        return
    
    # Test content balance analysis
    balance_results = test_content_balance_analysis()
    
    # Test clustering system
    clustering_results = test_clustering_system()
    
    # Test integration with quality assessor
    integration_results = test_integration_with_quality_assessor()
    
    # Summary
    logger.info("=== Phase 2 Test Summary ===")
    print(f"✓ Configuration loading: {'PASSED' if config_success else 'FAILED'}")
    print(f"✓ Content balance analysis: PASSED")
    print(f"✓ Clustering system: PASSED")
    print(f"✓ Quality assessor integration: PASSED")
    
    print(f"\nKey Results:")
    print(f"- Content balance score: {balance_results['distribution']['distribution_metrics']['balance_score']:.3f}")
    print(f"- Content gaps identified: {len(balance_results['gaps'])}")
    print(f"- Topic clusters created: {clustering_results['topic_clusters']['n_clusters']}")
    print(f"- Average flashcard quality: {sum(r['overall_score'] for r in integration_results['quality_results']) / len(integration_results['quality_results']):.3f}")
    
    logger.info("Phase 2 Implementation Tests Completed Successfully!")


if __name__ == "__main__":
    main()
