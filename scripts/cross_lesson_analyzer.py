#!/usr/bin/env python3
"""
Cross-Lesson Analyzer - Phase 1 Implementation
Analyzes content similarities between lessons and generates relationship mappings.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import itertools

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossLessonAnalyzer:
    """Analyze relationships and similarities between lessons."""
    
    def __init__(self, workspace_path: str = "/workspace"):
        self.workspace_path = Path(workspace_path)
        self.config_path = self.workspace_path / "config"
        
        # Load content index and embeddings
        self.content_index = {}
        self.semantic_embeddings = {}
        self.tfidf_vectorizer = None
        self.lesson_relationships = {}
        
        self._load_content_index()
    
    def _load_content_index(self):
        """Load the content index created by LessonContentIndexer."""
        try:
            # Load main content index
            index_path = self.config_path / "lesson_content_index.json"
            if index_path.exists():
                with open(index_path, 'r', encoding='utf-8') as f:
                    self.content_index = json.load(f)
                logger.info(f"Loaded content index with {len(self.content_index)} lessons")
            
            # Load semantic embeddings
            embeddings_path = self.config_path / "semantic_embeddings.pkl"
            if embeddings_path.exists():
                with open(embeddings_path, 'rb') as f:
                    self.semantic_embeddings = pickle.load(f)
                logger.info(f"Loaded semantic embeddings for {len(self.semantic_embeddings)} lessons")
            
            # Load TF-IDF vectorizer
            vectorizer_path = self.config_path / "tfidf_vectorizer.pkl"
            if vectorizer_path.exists():
                with open(vectorizer_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                logger.info("Loaded TF-IDF vectorizer")
        
        except Exception as e:
            logger.error(f"Error loading content index: {e}")
    
    def calculate_content_similarity(self, lesson1_id: str, lesson2_id: str) -> float:
        """Calculate content similarity between two lessons using TF-IDF vectors."""
        if lesson1_id not in self.content_index or lesson2_id not in self.content_index:
            return 0.0
        
        try:
            fingerprint1 = np.array(self.content_index[lesson1_id]['content_fingerprint'])
            fingerprint2 = np.array(self.content_index[lesson2_id]['content_fingerprint'])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([fingerprint1], [fingerprint2])[0][0]
            return float(similarity)
        
        except Exception as e:
            logger.error(f"Error calculating content similarity between {lesson1_id} and {lesson2_id}: {e}")
            return 0.0
    
    def calculate_semantic_similarity(self, lesson1_id: str, lesson2_id: str) -> float:
        """Calculate semantic similarity using OpenAI embeddings."""
        if (lesson1_id not in self.semantic_embeddings or 
            lesson2_id not in self.semantic_embeddings):
            return 0.0
        
        try:
            embedding1 = np.array(self.semantic_embeddings[lesson1_id])
            embedding2 = np.array(self.semantic_embeddings[lesson2_id])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
        
        except Exception as e:
            logger.error(f"Error calculating semantic similarity between {lesson1_id} and {lesson2_id}: {e}")
            return 0.0
    
    def calculate_concept_overlap(self, lesson1_id: str, lesson2_id: str) -> Dict[str, Any]:
        """Calculate concept overlap between two lessons."""
        if lesson1_id not in self.content_index or lesson2_id not in self.content_index:
            return {'overlap_score': 0.0, 'shared_concepts': [], 'jaccard_index': 0.0}
        
        try:
            concepts1 = set(self.content_index[lesson1_id]['key_concepts'])
            concepts2 = set(self.content_index[lesson2_id]['key_concepts'])
            
            # Find shared concepts
            shared_concepts = concepts1.intersection(concepts2)
            
            # Calculate Jaccard index
            union_concepts = concepts1.union(concepts2)
            jaccard_index = len(shared_concepts) / len(union_concepts) if union_concepts else 0.0
            
            # Calculate overlap score (ratio of shared concepts to smaller set)
            min_concepts = min(len(concepts1), len(concepts2))
            overlap_score = len(shared_concepts) / min_concepts if min_concepts > 0 else 0.0
            
            return {
                'overlap_score': overlap_score,
                'shared_concepts': list(shared_concepts),
                'jaccard_index': jaccard_index,
                'lesson1_unique': list(concepts1 - concepts2),
                'lesson2_unique': list(concepts2 - concepts1)
            }
        
        except Exception as e:
            logger.error(f"Error calculating concept overlap between {lesson1_id} and {lesson2_id}: {e}")
            return {'overlap_score': 0.0, 'shared_concepts': [], 'jaccard_index': 0.0}
    
    def detect_prerequisite_relationships(self) -> Dict[str, List[str]]:
        """Detect potential prerequisite relationships between lessons."""
        prerequisites = defaultdict(list)
        
        # Simple heuristic: lessons with higher concept overlap might have prerequisite relationships
        lesson_ids = list(self.content_index.keys())
        
        for lesson1_id, lesson2_id in itertools.combinations(lesson_ids, 2):
            concept_analysis = self.calculate_concept_overlap(lesson1_id, lesson2_id)
            
            if concept_analysis['overlap_score'] > 0.3:  # Threshold for significant overlap
                # Determine which lesson might be a prerequisite
                lesson1_concepts = len(self.content_index[lesson1_id]['key_concepts'])
                lesson2_concepts = len(self.content_index[lesson2_id]['key_concepts'])
                
                # The lesson with fewer concepts might be a prerequisite
                if lesson1_concepts < lesson2_concepts:
                    prerequisites[lesson2_id].append(lesson1_id)
                elif lesson2_concepts < lesson1_concepts:
                    prerequisites[lesson1_id].append(lesson2_id)
        
        return dict(prerequisites)
    
    def generate_lesson_relationship_graph(self) -> Dict[str, Any]:
        """Generate a comprehensive relationship graph between all lessons."""
        relationships = {}
        lesson_ids = list(self.content_index.keys())
        
        logger.info(f"Analyzing relationships between {len(lesson_ids)} lessons...")
        
        for lesson1_id, lesson2_id in itertools.combinations(lesson_ids, 2):
            # Calculate various similarity metrics
            content_similarity = self.calculate_content_similarity(lesson1_id, lesson2_id)
            semantic_similarity = self.calculate_semantic_similarity(lesson1_id, lesson2_id)
            concept_analysis = self.calculate_concept_overlap(lesson1_id, lesson2_id)
            
            # Determine relationship type and strength
            relationship_data = {
                'content_similarity': content_similarity,
                'semantic_similarity': semantic_similarity,
                'concept_overlap_score': concept_analysis['overlap_score'],
                'jaccard_index': concept_analysis['jaccard_index'],
                'shared_concepts': concept_analysis['shared_concepts'],
                'relationship_strength': self._calculate_relationship_strength(
                    content_similarity, semantic_similarity, concept_analysis['overlap_score']
                ),
                'relationship_type': self._determine_relationship_type(
                    content_similarity, semantic_similarity, concept_analysis
                )
            }
            
            # Store bidirectional relationships
            relationship_key = f"{lesson1_id}_{lesson2_id}"
            relationships[relationship_key] = relationship_data
        
        return relationships
    
    def _calculate_relationship_strength(self, content_sim: float, semantic_sim: float, concept_overlap: float) -> float:
        """Calculate overall relationship strength between two lessons."""
        # Weighted average of different similarity metrics
        weights = {
            'content': 0.3,
            'semantic': 0.4,
            'concepts': 0.3
        }
        
        strength = (
            weights['content'] * content_sim +
            weights['semantic'] * semantic_sim +
            weights['concepts'] * concept_overlap
        )
        
        return min(1.0, max(0.0, strength))  # Clamp between 0 and 1
    
    def _determine_relationship_type(self, content_sim: float, semantic_sim: float, concept_analysis: Dict) -> str:
        """Determine the type of relationship between two lessons."""
        overlap_score = concept_analysis['overlap_score']
        
        # Define thresholds
        high_threshold = 0.7
        medium_threshold = 0.4
        low_threshold = 0.2
        
        # Determine relationship type based on similarity scores
        max_similarity = max(content_sim, semantic_sim, overlap_score)
        
        if max_similarity >= high_threshold:
            return "closely_related"
        elif max_similarity >= medium_threshold:
            return "related"
        elif max_similarity >= low_threshold:
            return "somewhat_related"
        else:
            return "distantly_related"
    
    def identify_lesson_clusters(self, min_similarity: float = 0.5) -> List[List[str]]:
        """Identify clusters of related lessons."""
        lesson_ids = list(self.content_index.keys())
        clusters = []
        visited = set()
        
        for lesson_id in lesson_ids:
            if lesson_id in visited:
                continue
            
            # Start a new cluster
            cluster = [lesson_id]
            visited.add(lesson_id)
            
            # Find related lessons
            for other_lesson_id in lesson_ids:
                if other_lesson_id in visited:
                    continue
                
                # Check relationship strength
                content_sim = self.calculate_content_similarity(lesson_id, other_lesson_id)
                semantic_sim = self.calculate_semantic_similarity(lesson_id, other_lesson_id)
                concept_overlap = self.calculate_concept_overlap(lesson_id, other_lesson_id)['overlap_score']
                
                relationship_strength = self._calculate_relationship_strength(
                    content_sim, semantic_sim, concept_overlap
                )
                
                if relationship_strength >= min_similarity:
                    cluster.append(other_lesson_id)
                    visited.add(other_lesson_id)
            
            if len(cluster) > 1:  # Only include clusters with multiple lessons
                clusters.append(cluster)
        
        return clusters
    
    def generate_context_recommendations(self, target_lesson_id: str, max_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Generate context enhancement recommendations for a specific lesson."""
        if target_lesson_id not in self.content_index:
            return []
        
        recommendations = []
        
        for lesson_id in self.content_index.keys():
            if lesson_id == target_lesson_id:
                continue
            
            # Calculate relationship metrics
            content_sim = self.calculate_content_similarity(target_lesson_id, lesson_id)
            semantic_sim = self.calculate_semantic_similarity(target_lesson_id, lesson_id)
            concept_analysis = self.calculate_concept_overlap(target_lesson_id, lesson_id)
            
            relationship_strength = self._calculate_relationship_strength(
                content_sim, semantic_sim, concept_analysis['overlap_score']
            )
            
            if relationship_strength > 0.2:  # Minimum threshold for recommendations
                recommendations.append({
                    'lesson_id': lesson_id,
                    'relationship_strength': relationship_strength,
                    'content_similarity': content_sim,
                    'semantic_similarity': semantic_sim,
                    'concept_overlap': concept_analysis['overlap_score'],
                    'shared_concepts': concept_analysis['shared_concepts'],
                    'recommendation_reason': self._generate_recommendation_reason(
                        content_sim, semantic_sim, concept_analysis
                    )
                })
        
        # Sort by relationship strength and return top recommendations
        recommendations.sort(key=lambda x: x['relationship_strength'], reverse=True)
        return recommendations[:max_recommendations]
    
    def _generate_recommendation_reason(self, content_sim: float, semantic_sim: float, concept_analysis: Dict) -> str:
        """Generate a human-readable reason for the recommendation."""
        reasons = []
        
        if semantic_sim > 0.6:
            reasons.append("high semantic similarity")
        elif semantic_sim > 0.4:
            reasons.append("moderate semantic similarity")
        
        if content_sim > 0.6:
            reasons.append("similar content structure")
        elif content_sim > 0.4:
            reasons.append("related content patterns")
        
        if concept_analysis['overlap_score'] > 0.5:
            reasons.append(f"shares {len(concept_analysis['shared_concepts'])} key concepts")
        elif concept_analysis['overlap_score'] > 0.3:
            reasons.append("has overlapping concepts")
        
        if not reasons:
            reasons.append("general topical relation")
        
        return "; ".join(reasons)
    
    def analyze_all_relationships(self) -> Dict[str, Any]:
        """Perform comprehensive analysis of all lesson relationships."""
        logger.info("Starting comprehensive cross-lesson analysis...")
        
        # Generate relationship graph
        relationships = self.generate_lesson_relationship_graph()
        
        # Detect prerequisites
        prerequisites = self.detect_prerequisite_relationships()
        
        # Identify clusters
        clusters = self.identify_lesson_clusters()
        
        # Generate overall statistics
        lesson_count = len(self.content_index)
        relationship_count = len(relationships)
        
        # Calculate average similarities
        content_similarities = [r['content_similarity'] for r in relationships.values()]
        semantic_similarities = [r['semantic_similarity'] for r in relationships.values() if r['semantic_similarity'] > 0]
        concept_overlaps = [r['concept_overlap_score'] for r in relationships.values()]
        
        analysis_results = {
            'relationships': relationships,
            'prerequisites': prerequisites,
            'clusters': clusters,
            'statistics': {
                'total_lessons': lesson_count,
                'total_relationships': relationship_count,
                'average_content_similarity': np.mean(content_similarities) if content_similarities else 0,
                'average_semantic_similarity': np.mean(semantic_similarities) if semantic_similarities else 0,
                'average_concept_overlap': np.mean(concept_overlaps) if concept_overlaps else 0,
                'cluster_count': len(clusters)
            }
        }
        
        self.lesson_relationships = analysis_results
        return analysis_results
    
    def save_relationship_analysis(self):
        """Save the relationship analysis to files."""
        if not self.lesson_relationships:
            logger.warning("No relationship analysis to save")
            return
        
        # Save main analysis
        relationships_path = self.config_path / "lesson_relationships.json"
        with open(relationships_path, 'w', encoding='utf-8') as f:
            json.dump(self.lesson_relationships, f, indent=2)
        
        # Save simplified version for quick access
        simplified_path = self.config_path / "lesson_relationships_simplified.json"
        simplified = {
            'lesson_clusters': self.lesson_relationships['clusters'],
            'prerequisites': self.lesson_relationships['prerequisites'],
            'statistics': self.lesson_relationships['statistics']
        }
        with open(simplified_path, 'w', encoding='utf-8') as f:
            json.dump(simplified, f, indent=2)
        
        logger.info(f"Relationship analysis saved to {relationships_path}")
        logger.info(f"Simplified analysis saved to {simplified_path}")
    
    def load_relationship_analysis(self) -> bool:
        """Load existing relationship analysis."""
        try:
            relationships_path = self.config_path / "lesson_relationships.json"
            if relationships_path.exists():
                with open(relationships_path, 'r', encoding='utf-8') as f:
                    self.lesson_relationships = json.load(f)
                logger.info("Loaded existing relationship analysis")
                return True
        except Exception as e:
            logger.error(f"Error loading relationship analysis: {e}")
        
        return False


def main():
    """Main function to run the cross-lesson analyzer."""
    analyzer = CrossLessonAnalyzer()
    
    if not analyzer.content_index:
        print("No content index found. Please run lesson_content_indexer.py first.")
        return
    
    # Try to load existing analysis
    if not analyzer.load_relationship_analysis():
        print("No existing relationship analysis found, creating new analysis...")
        
        # Perform comprehensive analysis
        results = analyzer.analyze_all_relationships()
        
        # Save the analysis
        analyzer.save_relationship_analysis()
        
        # Print summary
        stats = results['statistics']
        print(f"\nCross-Lesson Analysis Complete!")
        print(f"Total lessons analyzed: {stats['total_lessons']}")
        print(f"Total relationships found: {stats['total_relationships']}")
        print(f"Average content similarity: {stats['average_content_similarity']:.3f}")
        print(f"Average semantic similarity: {stats['average_semantic_similarity']:.3f}")
        print(f"Average concept overlap: {stats['average_concept_overlap']:.3f}")
        print(f"Lesson clusters identified: {stats['cluster_count']}")
        
        # Show clusters
        if results['clusters']:
            print(f"\nLesson Clusters:")
            for i, cluster in enumerate(results['clusters'], 1):
                print(f"  Cluster {i}: {', '.join(cluster)}")
        
        # Show prerequisites
        if results['prerequisites']:
            print(f"\nPotential Prerequisites:")
            for lesson, prereqs in results['prerequisites'].items():
                print(f"  {lesson} <- {', '.join(prereqs)}")
    
    else:
        stats = analyzer.lesson_relationships['statistics']
        print(f"Loaded existing analysis with {stats['total_lessons']} lessons and {stats['total_relationships']} relationships")


if __name__ == "__main__":
    main()