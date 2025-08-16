#!/usr/bin/env python3
"""
Flashcard Clustering System

Provides clustering capabilities for organizing flashcards by topic, difficulty,
source, and other criteria to improve content organization and analysis.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass

@dataclass
class ClusterInfo:
    """Information about a flashcard cluster"""
    cluster_id: str
    name: str
    size: int
    flashcards: List[Dict[str, Any]]
    centroid: Dict[str, Any]
    characteristics: Dict[str, Any]

class FlashcardClusterer:
    """Clustering system for organizing flashcards"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the clusterer with configuration"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Topic keywords for clustering
        self.topic_keywords = {
            'operations': ['operation', 'operational', 'mission', 'tactical', 'strategic', 'deployment'],
            'communications': ['communication', 'radio', 'signal', 'transmission', 'frequency', 'network'],
            'equipment': ['equipment', 'system', 'device', 'tool', 'gear', 'weapon', 'hardware'],
            'procedures': ['procedure', 'protocol', 'process', 'method', 'technique', 'standard'],
            'safety': ['safety', 'security', 'protection', 'hazard', 'risk', 'prevention'],
            'leadership': ['leadership', 'command', 'supervision', 'management', 'authority', 'control'],
            'training': ['training', 'education', 'instruction', 'learning', 'development', 'course'],
            'logistics': ['logistics', 'supply', 'maintenance', 'support', 'transport', 'storage'],
            'intelligence': ['intelligence', 'surveillance', 'reconnaissance', 'analysis', 'information'],
            'personnel': ['personnel', 'staff', 'team', 'crew', 'member', 'individual']
        }
        
        # Difficulty level definitions
        self.difficulty_levels = {
            'basic': {
                'keywords': ['basic', 'fundamental', 'simple', 'elementary', 'introductory'],
                'complexity_threshold': 0.3
            },
            'intermediate': {
                'keywords': ['intermediate', 'moderate', 'standard', 'regular', 'normal'],
                'complexity_threshold': 0.7
            },
            'advanced': {
                'keywords': ['advanced', 'complex', 'sophisticated', 'expert', 'specialized'],
                'complexity_threshold': 1.0
            }
        }
    
    def cluster_flashcards_by_topic(self, flashcards: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Cluster flashcards by topic using keyword matching"""
        try:
            clusters = defaultdict(list)
            
            for flashcard in flashcards:
                # Handle both dict and dataclass objects
                if hasattr(flashcard, 'get'):
                    # It's a dict
                    topic = self._determine_topic(flashcard)
                    clusters[topic].append(flashcard)
                else:
                    # It's a dataclass, convert to dict
                    flashcard_dict = {
                        'term': getattr(flashcard, 'term', ''),
                        'definition': getattr(flashcard, 'definition', ''),
                        'quality_score': getattr(flashcard, 'quality_score', 0.0),
                        'difficulty_level': getattr(flashcard, 'difficulty_level', 'unknown'),
                        'topic': getattr(flashcard, 'topic', 'general')
                    }
                    topic = self._determine_topic(flashcard_dict)
                    clusters[topic].append(flashcard_dict)
            
            # Convert to regular dict and add metadata
            result = {}
            for topic, cards in clusters.items():
                result[topic] = {
                    'flashcards': cards,
                    'count': len(cards),
                    'percentage': (len(cards) / len(flashcards)) * 100 if flashcards else 0,
                    'avg_quality': self._calculate_average_quality(cards),
                    'difficulty_distribution': self._get_difficulty_distribution(cards)
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error clustering flashcards by topic: {e}")
            return {}
    
    def cluster_flashcards_by_difficulty(self, flashcards: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Cluster flashcards by difficulty level"""
        try:
            clusters = defaultdict(list)
            
            for flashcard in flashcards:
                difficulty = self._determine_difficulty(flashcard)
                clusters[difficulty].append(flashcard)
            
            # Convert to regular dict and add metadata
            result = {}
            for difficulty, cards in clusters.items():
                result[difficulty] = {
                    'flashcards': cards,
                    'count': len(cards),
                    'percentage': (len(cards) / len(flashcards)) * 100 if flashcards else 0,
                    'avg_quality': self._calculate_average_quality(cards),
                    'topic_distribution': self._get_topic_distribution(cards)
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error clustering flashcards by difficulty: {e}")
            return {}
    
    def cluster_flashcards_by_source(self, flashcards: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Cluster flashcards by source slide or document"""
        try:
            clusters = defaultdict(list)
            
            for flashcard in flashcards:
                source = self._determine_source(flashcard)
                clusters[source].append(flashcard)
            
            # Convert to regular dict and add metadata
            result = {}
            for source, cards in clusters.items():
                result[source] = {
                    'flashcards': cards,
                    'count': len(cards),
                    'percentage': (len(cards) / len(flashcards)) * 100 if flashcards else 0,
                    'avg_quality': self._calculate_average_quality(cards),
                    'topic_distribution': self._get_topic_distribution(cards),
                    'difficulty_distribution': self._get_difficulty_distribution(cards)
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error clustering flashcards by source: {e}")
            return {}
    
    def cluster_flashcards_by_quality(self, flashcards: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Cluster flashcards by quality level"""
        try:
            clusters = {
                'excellent': [],
                'good': [],
                'fair': [],
                'poor': []
            }
            
            for flashcard in flashcards:
                quality_score = float(flashcard.get('quality_score', 0))
                
                if quality_score >= 0.8:
                    clusters['excellent'].append(flashcard)
                elif quality_score >= 0.6:
                    clusters['good'].append(flashcard)
                elif quality_score >= 0.4:
                    clusters['fair'].append(flashcard)
                else:
                    clusters['poor'].append(flashcard)
            
            # Convert to result format
            result = {}
            for quality_level, cards in clusters.items():
                if cards:  # Only include non-empty clusters
                    result[quality_level] = {
                        'flashcards': cards,
                        'count': len(cards),
                        'percentage': (len(cards) / len(flashcards)) * 100 if flashcards else 0,
                        'avg_quality': self._calculate_average_quality(cards),
                        'topic_distribution': self._get_topic_distribution(cards),
                        'difficulty_distribution': self._get_difficulty_distribution(cards)
                    }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error clustering flashcards by quality: {e}")
            return {}
    
    def generate_cluster_summaries(self, clusters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summaries for clusters"""
        try:
            summaries = {}
            
            for cluster_name, cluster_data in clusters.items():
                flashcards = cluster_data.get('flashcards', [])
                
                if not flashcards:
                    continue
                
                # Calculate cluster characteristics
                characteristics = {
                    'total_flashcards': len(flashcards),
                    'average_quality_score': self._calculate_average_quality(flashcards),
                    'quality_range': self._calculate_quality_range(flashcards),
                    'topic_distribution': self._get_topic_distribution(flashcards),
                    'difficulty_distribution': self._get_difficulty_distribution(flashcards),
                    'common_terms': self._find_common_terms(flashcards),
                    'avg_definition_length': self._calculate_avg_definition_length(flashcards),
                    'duplicate_count': self._count_duplicates(flashcards)
                }
                
                # Generate recommendations
                recommendations = self._generate_cluster_recommendations(characteristics)
                
                summaries[cluster_name] = {
                    'characteristics': characteristics,
                    'recommendations': recommendations,
                    'strengths': self._identify_cluster_strengths(characteristics),
                    'weaknesses': self._identify_cluster_weaknesses(characteristics)
                }
            
            return summaries
            
        except Exception as e:
            self.logger.error(f"Error generating cluster summaries: {e}")
            return {}
    
    def _determine_topic(self, flashcard: Dict[str, Any]) -> str:
        """Determine the topic of a flashcard"""
        term = flashcard.get('term', '').lower()
        definition = flashcard.get('definition', '').lower()
        
        text = f"{term} {definition}"
        
        # Count keyword matches for each topic
        topic_scores = {}
        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                topic_scores[topic] = score
        
        # Return the topic with the highest score
        if topic_scores:
            return max(topic_scores.items(), key=lambda x: x[1])[0]
        
        return 'general'
    
    def _determine_difficulty(self, flashcard: Dict[str, Any]) -> str:
        """Determine the difficulty level of a flashcard"""
        # Use existing difficulty level if available
        if 'difficulty_level' in flashcard:
            difficulty = flashcard['difficulty_level']
            if difficulty in self.difficulty_levels:
                return difficulty
        
        # Calculate difficulty based on content
        term = flashcard.get('term', '').lower()
        definition = flashcard.get('definition', '').lower()
        
        # Count complex words and military terms
        complex_words = sum(1 for word in term.split() if len(word) > 8)
        military_terms = sum(1 for word in term.split() if word in self._get_military_terms())
        
        complexity_score = (complex_words + military_terms) / max(1, len(term.split()))
        
        # Determine difficulty level
        if complexity_score <= 0.3:
            return 'basic'
        elif complexity_score <= 0.7:
            return 'intermediate'
        else:
            return 'advanced'
    
    def _determine_source(self, flashcard: Dict[str, Any]) -> str:
        """Determine the source of a flashcard"""
        source_slide = flashcard.get('source_slide')
        if source_slide is not None:
            return f"slide_{source_slide}"
        
        # Try to extract from other metadata
        source = flashcard.get('source', 'unknown')
        return str(source)
    
    def _calculate_average_quality(self, flashcards: List[Dict[str, Any]]) -> float:
        """Calculate average quality score for a list of flashcards"""
        if not flashcards:
            return 0.0
        
        total_score = sum(float(fc.get('quality_score', 0)) for fc in flashcards)
        return round(total_score / len(flashcards), 3)
    
    def _calculate_quality_range(self, flashcards: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate quality score range for flashcards"""
        if not flashcards:
            return {'min': 0.0, 'max': 0.0, 'range': 0.0}
        
        scores = [float(fc.get('quality_score', 0)) for fc in flashcards]
        min_score = min(scores)
        max_score = max(scores)
        
        return {
            'min': round(min_score, 3),
            'max': round(max_score, 3),
            'range': round(max_score - min_score, 3)
        }
    
    def _get_topic_distribution(self, flashcards: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get topic distribution for flashcards"""
        topics = [self._determine_topic(fc) for fc in flashcards]
        return dict(Counter(topics))
    
    def _get_difficulty_distribution(self, flashcards: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get difficulty distribution for flashcards"""
        difficulties = [self._determine_difficulty(fc) for fc in flashcards]
        return dict(Counter(difficulties))
    
    def _find_common_terms(self, flashcards: List[Dict[str, Any]], top_n: int = 5) -> List[str]:
        """Find common terms in a cluster"""
        all_terms = []
        for fc in flashcards:
            term = fc.get('term', '').lower()
            # Extract individual words
            words = re.findall(r'\b\w+\b', term)
            all_terms.extend(words)
        
        # Count and return most common
        term_counts = Counter(all_terms)
        return [term for term, count in term_counts.most_common(top_n)]
    
    def _calculate_avg_definition_length(self, flashcards: List[Dict[str, Any]]) -> float:
        """Calculate average definition length"""
        if not flashcards:
            return 0.0
        
        total_length = sum(len(fc.get('definition', '')) for fc in flashcards)
        return round(total_length / len(flashcards), 1)
    
    def _count_duplicates(self, flashcards: List[Dict[str, Any]]) -> int:
        """Count duplicate flashcards in a cluster"""
        duplicate_count = 0
        seen_terms = set()
        
        for fc in flashcards:
            term = fc.get('term', '').strip().lower()
            if term in seen_terms:
                duplicate_count += 1
            else:
                seen_terms.add(term)
        
        return duplicate_count
    
    def _get_military_terms(self) -> set:
        """Get set of military terms for complexity assessment"""
        military_terms = set()
        for keywords in self.topic_keywords.values():
            military_terms.update(keywords)
        return military_terms
    
    def _generate_cluster_recommendations(self, characteristics: Dict[str, Any]) -> List[str]:
        """Generate recommendations for a cluster"""
        recommendations = []
        
        avg_quality = characteristics.get('average_quality_score', 0)
        duplicate_count = characteristics.get('duplicate_count', 0)
        total_cards = characteristics.get('total_flashcards', 0)
        
        if avg_quality < 0.6:
            recommendations.append("Consider improving flashcard quality in this cluster")
        
        if duplicate_count > total_cards * 0.1:  # More than 10% duplicates
            recommendations.append("High number of duplicates - consider merging or removing")
        
        if total_cards < 3:
            recommendations.append("Small cluster - consider expanding content coverage")
        
        if not recommendations:
            recommendations.append("Cluster quality is good")
        
        return recommendations
    
    def _identify_cluster_strengths(self, characteristics: Dict[str, Any]) -> List[str]:
        """Identify strengths of a cluster"""
        strengths = []
        
        avg_quality = characteristics.get('average_quality_score', 0)
        duplicate_count = characteristics.get('duplicate_count', 0)
        total_cards = characteristics.get('total_flashcards', 0)
        
        if avg_quality >= 0.8:
            strengths.append("High average quality score")
        
        if duplicate_count == 0:
            strengths.append("No duplicate flashcards")
        
        if total_cards >= 10:
            strengths.append("Good content coverage")
        
        return strengths
    
    def _identify_cluster_weaknesses(self, characteristics: Dict[str, Any]) -> List[str]:
        """Identify weaknesses of a cluster"""
        weaknesses = []
        
        avg_quality = characteristics.get('average_quality_score', 0)
        duplicate_count = characteristics.get('duplicate_count', 0)
        total_cards = characteristics.get('total_flashcards', 0)
        
        if avg_quality < 0.6:
            weaknesses.append("Low average quality score")
        
        if duplicate_count > 0:
            weaknesses.append(f"{duplicate_count} duplicate flashcards")
        
        if total_cards < 3:
            weaknesses.append("Limited content coverage")
        
        return weaknesses
    
    def get_clustering_statistics(self, clusters: Dict[str, Any]) -> Dict[str, Any]:
        """Get overall statistics for clustering results"""
        try:
            total_clusters = len(clusters)
            total_flashcards = sum(cluster.get('count', 0) for cluster in clusters.values())
            
            # Calculate distribution statistics
            cluster_sizes = [cluster.get('count', 0) for cluster in clusters.values()]
            avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
            
            # Quality statistics
            all_qualities = []
            for cluster in clusters.values():
                all_qualities.append(cluster.get('avg_quality', 0))
            
            avg_overall_quality = sum(all_qualities) / len(all_qualities) if all_qualities else 0
            
            return {
                'total_clusters': total_clusters,
                'total_flashcards': total_flashcards,
                'average_cluster_size': round(avg_cluster_size, 1),
                'average_overall_quality': round(avg_overall_quality, 3),
                'largest_cluster': max(cluster_sizes) if cluster_sizes else 0,
                'smallest_cluster': min(cluster_sizes) if cluster_sizes else 0,
                'cluster_size_distribution': dict(Counter(cluster_sizes))
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating clustering statistics: {e}")
            return {}
