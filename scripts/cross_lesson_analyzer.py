#!/usr/bin/env python3
"""
Cross-Lesson Analyzer for Enhanced Context System

Performs relationship analysis and similarity detection between lessons
to enhance flashcard generation with cross-lesson context.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
except ImportError:
    print("WARNING: scikit-learn and numpy required for advanced analysis")
    np = None
    TfidfVectorizer = None
    cosine_similarity = None
    KMeans = None

try:
    from lesson_content_indexer import LessonContentIndexer
except ImportError:
    print("WARNING: lesson_content_indexer not found. Limited functionality.")
    LessonContentIndexer = None

class CrossLessonAnalyzer:
    """Analyzes relationships and similarities between lessons"""
    
    def __init__(self, lessons_dir: str = "lessons", config_dir: str = "config"):
        self.lessons_dir = Path(lessons_dir)
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize content indexer
        if LessonContentIndexer:
            self.indexer = LessonContentIndexer(lessons_dir, config_dir)
        else:
            self.indexer = None
            print("WARNING: Content indexer not available")
        
        # Analysis data
        self.lesson_embeddings = {}
        self.similarity_matrix = {}
        self.concept_clusters = {}
        self.cross_references = defaultdict(list)
        
        # Load existing analysis if available
        self.load_analysis()
    
    def load_analysis(self) -> bool:
        """Load existing cross-lesson analysis from disk"""
        analysis_file = self.config_dir / "cross_lesson_analysis.json"
        if analysis_file.exists():
            try:
                with open(analysis_file, 'r') as f:
                    data = json.load(f)
                    self.similarity_matrix = data.get('similarity_matrix', {})
                    self.concept_clusters = data.get('concept_clusters', {})
                    self.cross_references = defaultdict(list, data.get('cross_references', {}))
                print(f"✓ Loaded existing cross-lesson analysis")
                return True
            except Exception as e:
                print(f"WARNING: Could not load existing analysis: {e}")
        return False
    
    def save_analysis(self) -> bool:
        """Save cross-lesson analysis to disk"""
        try:
            analysis_file = self.config_dir / "cross_lesson_analysis.json"
            data = {
                'similarity_matrix': self.similarity_matrix,
                'concept_clusters': self.concept_clusters,
                'cross_references': dict(self.cross_references),
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            }
            with open(analysis_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"✓ Saved cross-lesson analysis")
            return True
        except Exception as e:
            print(f"ERROR: Could not save analysis: {e}")
            return False
    
    def analyze_all_lessons(self) -> Dict[str, Any]:
        """Perform comprehensive cross-lesson analysis"""
        print("🔍 Starting cross-lesson analysis...")
        
        # Ensure lessons are indexed
        if self.indexer:
            self.indexer.index_all_lessons()
        
        # Get all lessons
        lessons = self._get_available_lessons()
        if not lessons:
            print("ERROR: No lessons found for analysis")
            return {}
        
        print(f"📚 Analyzing {len(lessons)} lessons")
        
        # Perform different types of analysis
        results = {}
        
        # 1. Content similarity analysis
        print("⚡ Analyzing content similarity...")
        results['similarity_analysis'] = self._analyze_content_similarity(lessons)
        
        # 2. Concept clustering
        print("🎯 Performing concept clustering...")
        results['concept_clustering'] = self._perform_concept_clustering(lessons)
        
        # 3. Cross-reference detection
        print("🔗 Detecting cross-references...")
        results['cross_references'] = self._detect_cross_references(lessons)
        
        # 4. Relationship mapping
        print("🗺️ Mapping lesson relationships...")
        results['relationship_mapping'] = self._map_lesson_relationships(lessons)
        
        # Save results
        self.save_analysis()
        
        # Generate summary
        summary = {
            'total_lessons': len(lessons),
            'similarity_pairs': len(self.similarity_matrix),
            'concept_clusters': len(self.concept_clusters),
            'cross_references': sum(len(refs) for refs in self.cross_references.values()),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        results['summary'] = summary
        print(f"📊 Analysis complete: {summary}")
        return results
    
    def _get_available_lessons(self) -> List[str]:
        """Get list of available lessons"""
        lessons = []
        if self.lessons_dir.exists():
            for lesson_dir in self.lessons_dir.iterdir():
                if lesson_dir.is_dir() and not lesson_dir.name.startswith('.'):
                    lessons.append(lesson_dir.name)
        return sorted(lessons)
    
    def _analyze_content_similarity(self, lessons: List[str]) -> Dict[str, Any]:
        """Analyze content similarity between lessons"""
        similarity_results = {}
        
        if not self.indexer:
            print("WARNING: No indexer available for similarity analysis")
            return similarity_results
        
        # Get lesson content for analysis
        lesson_contents = {}
        for lesson in lessons:
            content = self._extract_lesson_text_content(lesson)
            if content:
                lesson_contents[lesson] = content
        
        if len(lesson_contents) < 2:
            print("WARNING: Insufficient content for similarity analysis")
            return similarity_results
        
        # Perform TF-IDF analysis if available
        if TfidfVectorizer and cosine_similarity:
            try:
                vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                
                lesson_names = list(lesson_contents.keys())
                documents = [lesson_contents[lesson] for lesson in lesson_names]
                
                tfidf_matrix = vectorizer.fit_transform(documents)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Store similarity results
                for i, lesson1 in enumerate(lesson_names):
                    self.similarity_matrix[lesson1] = {}
                    for j, lesson2 in enumerate(lesson_names):
                        if i != j:
                            similarity_score = float(similarity_matrix[i][j])
                            self.similarity_matrix[lesson1][lesson2] = similarity_score
                
                similarity_results['method'] = 'tfidf_cosine'
                similarity_results['lessons_analyzed'] = len(lesson_names)
                similarity_results['feature_count'] = tfidf_matrix.shape[1]
                
            except Exception as e:
                print(f"WARNING: TF-IDF analysis failed: {e}")
        
        # Fallback to simple keyword similarity
        else:
            for i, lesson1 in enumerate(lessons):
                self.similarity_matrix[lesson1] = {}
                for j, lesson2 in enumerate(lessons):
                    if i != j and self.indexer:
                        # Use indexer's similarity calculation
                        similarity = self.indexer._calculate_lesson_similarity(lesson1, lesson2)
                        self.similarity_matrix[lesson1][lesson2] = similarity
            
            similarity_results['method'] = 'keyword_jaccard'
            similarity_results['lessons_analyzed'] = len(lessons)
        
        return similarity_results
    
    def _extract_lesson_text_content(self, lesson_name: str) -> str:
        """Extract text content from a lesson for analysis"""
        if not self.indexer or lesson_name not in self.indexer.lesson_index:
            return ""
        
        lesson_data = self.indexer.lesson_index[lesson_name]
        
        # Combine various text sources
        text_parts = []
        
        # Add lesson name (processed)
        text_parts.append(lesson_name.replace('_', ' '))
        
        # Add keywords and topics
        keywords = self.indexer.topic_keywords.get(lesson_name, set())
        text_parts.extend(list(keywords))
        
        # Add any processed content text (if available)
        processed_content = lesson_data.get('processed_content', {})
        for filename, content_info in processed_content.items():
            if content_info.get('type') == 'flashcards':
                # Add filename as context
                text_parts.append(filename.replace('_', ' ').replace('.json', ''))
        
        return ' '.join(text_parts)
    
    def _perform_concept_clustering(self, lessons: List[str]) -> Dict[str, Any]:
        """Perform concept clustering on lessons"""
        clustering_results = {}
        
        if not lessons or not self.indexer:
            return clustering_results
        
        # Collect all concepts and keywords
        all_concepts = set()
        lesson_concepts = {}
        
        for lesson in lessons:
            concepts = self.indexer.topic_keywords.get(lesson, set())
            lesson_concepts[lesson] = concepts
            all_concepts.update(concepts)
        
        if len(all_concepts) < 3:
            print("WARNING: Insufficient concepts for clustering")
            return clustering_results
        
        # Perform simple clustering based on shared concepts
        concept_to_lessons = defaultdict(list)
        for lesson, concepts in lesson_concepts.items():
            for concept in concepts:
                concept_to_lessons[concept].append(lesson)
        
        # Find concept clusters (concepts shared by multiple lessons)
        clusters = {}
        cluster_id = 0
        
        for concept, lesson_list in concept_to_lessons.items():
            if len(lesson_list) > 1:  # Shared concept
                cluster_name = f"cluster_{cluster_id}"
                clusters[cluster_name] = {
                    'concept': concept,
                    'lessons': lesson_list,
                    'strength': len(lesson_list)
                }
                cluster_id += 1
        
        self.concept_clusters = clusters
        
        clustering_results = {
            'total_concepts': len(all_concepts),
            'shared_concepts': len(clusters),
            'clusters': clusters
        }
        
        return clustering_results
    
    def _detect_cross_references(self, lessons: List[str]) -> Dict[str, Any]:
        """Detect cross-references between lessons"""
        cross_ref_results = {}
        
        if not self.indexer:
            return cross_ref_results
        
        # Clear existing cross-references
        self.cross_references.clear()
        
        # Detect references based on similarity thresholds
        similarity_threshold = 0.2
        
        for lesson1 in lessons:
            similarities = self.similarity_matrix.get(lesson1, {})
            
            for lesson2, similarity in similarities.items():
                if similarity > similarity_threshold:
                    # Determine reference type
                    ref_type = self._determine_reference_type(lesson1, lesson2, similarity)
                    
                    self.cross_references[lesson1].append({
                        'target_lesson': lesson2,
                        'similarity': similarity,
                        'reference_type': ref_type,
                        'strength': self._calculate_reference_strength(lesson1, lesson2)
                    })
        
        # Sort references by strength
        for lesson in self.cross_references:
            self.cross_references[lesson].sort(
                key=lambda x: x['strength'], 
                reverse=True
            )
        
        cross_ref_results = {
            'lessons_with_references': len(self.cross_references),
            'total_references': sum(len(refs) for refs in self.cross_references.values()),
            'average_references_per_lesson': sum(len(refs) for refs in self.cross_references.values()) / max(len(lessons), 1)
        }
        
        return cross_ref_results
    
    def _determine_reference_type(self, lesson1: str, lesson2: str, similarity: float) -> str:
        """Determine the type of cross-reference between lessons"""
        if similarity > 0.5:
            return 'strong_topical'
        elif similarity > 0.3:
            return 'moderate_topical'
        elif self._check_procedural_relationship(lesson1, lesson2):
            return 'procedural'
        elif self._check_hierarchical_relationship(lesson1, lesson2):
            return 'hierarchical'
        else:
            return 'weak_topical'
    
    def _check_procedural_relationship(self, lesson1: str, lesson2: str) -> bool:
        """Check if lessons have a procedural relationship"""
        procedural_keywords = ['tlp', 'procedure', 'step', 'process', 'conduct']
        
        lesson1_lower = lesson1.lower()
        lesson2_lower = lesson2.lower()
        
        return any(keyword in lesson1_lower and keyword in lesson2_lower 
                  for keyword in procedural_keywords)
    
    def _check_hierarchical_relationship(self, lesson1: str, lesson2: str) -> bool:
        """Check if lessons have a hierarchical relationship"""
        # Simple heuristic: if one lesson name contains the other
        return (lesson1.lower() in lesson2.lower() or 
                lesson2.lower() in lesson1.lower())
    
    def _calculate_reference_strength(self, lesson1: str, lesson2: str) -> float:
        """Calculate the strength of cross-reference between lessons"""
        base_similarity = self.similarity_matrix.get(lesson1, {}).get(lesson2, 0.0)
        
        # Add bonuses for different factors
        strength = base_similarity
        
        # Bonus for shared concepts
        if self.indexer:
            concepts1 = self.indexer.topic_keywords.get(lesson1, set())
            concepts2 = self.indexer.topic_keywords.get(lesson2, set())
            shared_concepts = len(concepts1.intersection(concepts2))
            strength += shared_concepts * 0.1
        
        # Bonus for procedural relationships
        if self._check_procedural_relationship(lesson1, lesson2):
            strength += 0.2
        
        # Bonus for hierarchical relationships
        if self._check_hierarchical_relationship(lesson1, lesson2):
            strength += 0.1
        
        return min(strength, 1.0)  # Cap at 1.0
    
    def _map_lesson_relationships(self, lessons: List[str]) -> Dict[str, Any]:
        """Map relationships between lessons"""
        relationship_map = {}
        
        for lesson in lessons:
            relationships = []
            
            # Get cross-references for this lesson
            refs = self.cross_references.get(lesson, [])
            
            for ref in refs[:5]:  # Top 5 related lessons
                relationships.append({
                    'related_lesson': ref['target_lesson'],
                    'relationship_strength': ref['strength'],
                    'relationship_type': ref['reference_type'],
                    'similarity_score': ref['similarity']
                })
            
            relationship_map[lesson] = relationships
        
        return relationship_map
    
    def get_context_for_lesson(self, lesson_name: str, max_context_lessons: int = 3) -> Dict[str, Any]:
        """Get contextual information for a specific lesson"""
        if lesson_name not in self._get_available_lessons():
            return {}
        
        context = {
            'target_lesson': lesson_name,
            'related_lessons': [],
            'shared_concepts': [],
            'context_strength': 0.0
        }
        
        # Get related lessons from cross-references
        refs = self.cross_references.get(lesson_name, [])
        context['related_lessons'] = refs[:max_context_lessons]
        
        # Calculate overall context strength
        if refs:
            total_strength = sum(ref['strength'] for ref in refs[:max_context_lessons])
            context['context_strength'] = total_strength / max_context_lessons
        
        # Get shared concepts
        if self.indexer:
            lesson_concepts = self.indexer.topic_keywords.get(lesson_name, set())
            for ref in refs[:max_context_lessons]:
                related_lesson = ref['target_lesson']
                related_concepts = self.indexer.topic_keywords.get(related_lesson, set())
                shared = lesson_concepts.intersection(related_concepts)
                if shared:
                    context['shared_concepts'].extend(list(shared))
        
        # Remove duplicates
        context['shared_concepts'] = list(set(context['shared_concepts']))
        
        return context
    
    def generate_context_recommendations(self, lesson_name: str) -> List[Dict[str, Any]]:
        """Generate recommendations for using cross-lesson context"""
        recommendations = []
        
        context = self.get_context_for_lesson(lesson_name)
        
        if not context['related_lessons']:
            recommendations.append({
                'type': 'warning',
                'message': f"No related lessons found for {lesson_name}",
                'action': 'Consider reviewing lesson content and running analysis again'
            })
            return recommendations
        
        # High context strength recommendation
        if context['context_strength'] > 0.5:
            recommendations.append({
                'type': 'high_context',
                'message': f"Strong cross-lesson context available for {lesson_name}",
                'action': 'Include context from related lessons in flashcard generation',
                'related_lessons': [ref['related_lesson'] for ref in context['related_lessons']]
            })
        
        # Shared concepts recommendation
        if len(context['shared_concepts']) > 3:
            recommendations.append({
                'type': 'shared_concepts',
                'message': f"Multiple shared concepts found: {', '.join(context['shared_concepts'][:5])}",
                'action': 'Use shared concepts to create cross-referencing flashcards',
                'concepts': context['shared_concepts']
            })
        
        # Procedural relationship recommendation
        procedural_refs = [ref for ref in context['related_lessons'] 
                          if ref['reference_type'] == 'procedural']
        if procedural_refs:
            recommendations.append({
                'type': 'procedural',
                'message': f"Procedural relationships found with {len(procedural_refs)} lessons",
                'action': 'Create sequential or step-based flashcards',
                'related_lessons': [ref['related_lesson'] for ref in procedural_refs]
            })
        
        return recommendations

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Analyze cross-lesson relationships and similarities")
    parser.add_argument("--lessons-dir", default="lessons", help="Directory containing lessons")
    parser.add_argument("--config-dir", default="config", help="Directory for configuration and output")
    parser.add_argument("--analyze", action="store_true", help="Perform full analysis")
    parser.add_argument("--context", help="Get context for specific lesson")
    parser.add_argument("--recommendations", help="Get recommendations for specific lesson")
    parser.add_argument("--similarity", nargs=2, metavar=('LESSON1', 'LESSON2'), help="Get similarity between two lessons")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    analyzer = CrossLessonAnalyzer(args.lessons_dir, args.config_dir)
    
    if args.analyze:
        results = analyzer.analyze_all_lessons()
        if args.verbose:
            print("\n📊 Analysis Results:")
            print(json.dumps(results, indent=2, default=str))
    
    elif args.context:
        context = analyzer.get_context_for_lesson(args.context)
        if context:
            print(f"\n🔍 Context for {args.context}:")
            print(json.dumps(context, indent=2, default=str))
        else:
            print(f"No context found for lesson: {args.context}")
    
    elif args.recommendations:
        recommendations = analyzer.generate_context_recommendations(args.recommendations)
        if recommendations:
            print(f"\n💡 Recommendations for {args.recommendations}:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['type'].upper()}: {rec['message']}")
                print(f"   Action: {rec['action']}")
                if 'related_lessons' in rec:
                    print(f"   Related: {', '.join(rec['related_lessons'])}")
                print()
        else:
            print(f"No recommendations generated for: {args.recommendations}")
    
    elif args.similarity:
        lesson1, lesson2 = args.similarity
        similarity = analyzer.similarity_matrix.get(lesson1, {}).get(lesson2, 0.0)
        print(f"Similarity between '{lesson1}' and '{lesson2}': {similarity:.3f}")
    
    else:
        print("Use --analyze to perform full analysis, or --help for more options")

if __name__ == "__main__":
    main()