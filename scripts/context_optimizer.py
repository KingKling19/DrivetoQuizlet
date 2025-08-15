#!/usr/bin/env python3
"""
Context Optimizer for Cross-Lesson Context System

Provides context optimization and quality assessment algorithms for 
adaptive context selection in flashcard generation.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict
import math

try:
    from lesson_content_indexer import LessonContentIndexer
    from cross_lesson_analyzer import CrossLessonAnalyzer
except ImportError:
    print("WARNING: Required modules not found. Limited functionality.")
    LessonContentIndexer = None
    CrossLessonAnalyzer = None

class ContextOptimizer:
    """Optimizes cross-lesson context selection and quality"""
    
    def __init__(self, lessons_dir: str = "lessons", config_dir: str = "config"):
        self.lessons_dir = Path(lessons_dir)
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize components
        if LessonContentIndexer and CrossLessonAnalyzer:
            self.indexer = LessonContentIndexer(lessons_dir, config_dir)
            self.analyzer = CrossLessonAnalyzer(lessons_dir, config_dir)
        else:
            self.indexer = None
            self.analyzer = None
            print("WARNING: Core components not available")
        
        # Optimization parameters
        self.optimization_config = {
            'semantic_weight': 0.4,
            'concept_overlap_weight': 0.3,
            'relationship_strength_weight': 0.2,
            'content_freshness_weight': 0.1,
            'max_context_lessons': 5,
            'min_quality_threshold': 0.3,
            'adaptive_threshold': True
        }
        
        # Quality metrics
        self.context_quality_history = {}
        self.optimization_metrics = {}
        
        # Load existing data
        self.load_optimization_data()
    
    def load_optimization_data(self) -> bool:
        """Load existing optimization data from disk"""
        opt_file = self.config_dir / "context_optimization.json"
        if opt_file.exists():
            try:
                with open(opt_file, 'r') as f:
                    data = json.load(f)
                    self.optimization_config.update(data.get('config', {}))
                    self.context_quality_history = data.get('quality_history', {})
                    self.optimization_metrics = data.get('metrics', {})
                print(f"✓ Loaded optimization data")
                return True
            except Exception as e:
                print(f"WARNING: Could not load optimization data: {e}")
        return False
    
    def save_optimization_data(self) -> bool:
        """Save optimization data to disk"""
        try:
            opt_file = self.config_dir / "context_optimization.json"
            data = {
                'config': self.optimization_config,
                'quality_history': self.context_quality_history,
                'metrics': self.optimization_metrics,
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            }
            with open(opt_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"✓ Saved optimization data")
            return True
        except Exception as e:
            print(f"ERROR: Could not save optimization data: {e}")
            return False
    
    def optimize_context_for_lesson(self, lesson_name: str, 
                                   target_flashcard_count: int = None) -> Dict[str, Any]:
        """Optimize context selection for a specific lesson"""
        if not self.analyzer:
            return {'error': 'Analyzer not available'}
        
        print(f"🎯 Optimizing context for lesson: {lesson_name}")
        
        # Get initial context
        raw_context = self.analyzer.get_context_for_lesson(lesson_name)
        if not raw_context['related_lessons']:
            return {
                'lesson': lesson_name,
                'optimized_context': [],
                'quality_score': 0.0,
                'recommendations': ['No related lessons found']
            }
        
        # Apply optimization algorithms
        optimized_context = self._apply_context_optimization(lesson_name, raw_context)
        
        # Assess quality
        quality_assessment = self._assess_context_quality(lesson_name, optimized_context)
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(
            lesson_name, optimized_context, quality_assessment
        )
        
        # Store quality history for learning
        self._update_quality_history(lesson_name, quality_assessment)
        
        result = {
            'lesson': lesson_name,
            'optimized_context': optimized_context,
            'quality_score': quality_assessment['overall_quality'],
            'quality_breakdown': quality_assessment,
            'recommendations': recommendations,
            'optimization_timestamp': datetime.now().isoformat()
        }
        
        print(f"✓ Context optimization complete. Quality score: {quality_assessment['overall_quality']:.3f}")
        return result
    
    def _apply_context_optimization(self, lesson_name: str, 
                                   raw_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply multi-factor optimization to context selection"""
        related_lessons = raw_context['related_lessons']
        
        # Calculate optimization scores for each related lesson
        optimized_lessons = []
        
        for rel_lesson in related_lessons:
            opt_score = self._calculate_optimization_score(
                lesson_name, rel_lesson, raw_context
            )
            
            optimized_lesson = {
                'lesson_name': rel_lesson['target_lesson'],
                'original_similarity': rel_lesson['similarity'],
                'original_strength': rel_lesson['strength'],
                'optimization_score': opt_score,
                'context_factors': self._analyze_context_factors(lesson_name, rel_lesson),
                'recommended_usage': self._determine_usage_recommendation(opt_score)
            }
            
            optimized_lessons.append(optimized_lesson)
        
        # Sort by optimization score and apply threshold
        optimized_lessons.sort(key=lambda x: x['optimization_score'], reverse=True)
        
        # Apply adaptive threshold
        threshold = self._calculate_adaptive_threshold(optimized_lessons)
        
        # Filter by threshold and max count
        filtered_lessons = [
            lesson for lesson in optimized_lessons 
            if lesson['optimization_score'] >= threshold
        ]
        
        max_lessons = self.optimization_config['max_context_lessons']
        return filtered_lessons[:max_lessons]
    
    def _calculate_optimization_score(self, target_lesson: str, 
                                    related_lesson: Dict[str, Any],
                                    raw_context: Dict[str, Any]) -> float:
        """Calculate multi-factor optimization score"""
        
        # Factor 1: Semantic similarity
        semantic_score = related_lesson['similarity']
        
        # Factor 2: Concept overlap
        concept_score = self._calculate_concept_overlap_score(
            target_lesson, related_lesson['target_lesson']
        )
        
        # Factor 3: Relationship strength
        relationship_score = related_lesson['strength']
        
        # Factor 4: Content freshness (how recently content was updated)
        freshness_score = self._calculate_content_freshness_score(
            related_lesson['target_lesson']
        )
        
        # Weighted combination
        config = self.optimization_config
        optimization_score = (
            semantic_score * config['semantic_weight'] +
            concept_score * config['concept_overlap_weight'] +
            relationship_score * config['relationship_strength_weight'] +
            freshness_score * config['content_freshness_weight']
        )
        
        return min(optimization_score, 1.0)  # Cap at 1.0
    
    def _calculate_concept_overlap_score(self, lesson1: str, lesson2: str) -> float:
        """Calculate concept overlap score between lessons"""
        if not self.indexer:
            return 0.0
        
        concepts1 = self.indexer.topic_keywords.get(lesson1, set())
        concepts2 = self.indexer.topic_keywords.get(lesson2, set())
        
        if not concepts1 or not concepts2:
            return 0.0
        
        # Jaccard similarity for concepts
        intersection = len(concepts1.intersection(concepts2))
        union = len(concepts1.union(concepts2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_content_freshness_score(self, lesson_name: str) -> float:
        """Calculate content freshness score based on modification dates"""
        if not self.indexer or lesson_name not in self.indexer.lesson_index:
            return 0.5  # Default neutral score
        
        lesson_data = self.indexer.lesson_index[lesson_name]
        
        # Get the most recent modification time from any content source
        latest_mod_time = 0
        
        for content_type in ['presentations', 'notes', 'audio']:
            content_list = lesson_data.get(content_type, [])
            for content_item in content_list:
                mod_time = content_item.get('modified', 0)
                latest_mod_time = max(latest_mod_time, mod_time)
        
        if latest_mod_time == 0:
            return 0.5  # No modification time available
        
        # Calculate freshness based on days since modification
        current_time = datetime.now().timestamp()
        days_since_mod = (current_time - latest_mod_time) / (24 * 60 * 60)
        
        # Exponential decay: fresher content gets higher scores
        freshness_score = math.exp(-days_since_mod / 30)  # 30-day half-life
        
        return min(freshness_score, 1.0)
    
    def _analyze_context_factors(self, target_lesson: str, 
                                related_lesson: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context factors for a related lesson"""
        factors = {
            'similarity_level': self._categorize_similarity(related_lesson['similarity']),
            'relationship_type': related_lesson.get('reference_type', 'unknown'),
            'strength_level': self._categorize_strength(related_lesson['strength']),
            'content_overlap': self._calculate_concept_overlap_score(
                target_lesson, related_lesson['target_lesson']
            )
        }
        
        return factors
    
    def _categorize_similarity(self, similarity: float) -> str:
        """Categorize similarity score into levels"""
        if similarity > 0.7:
            return 'very_high'
        elif similarity > 0.5:
            return 'high'
        elif similarity > 0.3:
            return 'moderate'
        elif similarity > 0.1:
            return 'low'
        else:
            return 'very_low'
    
    def _categorize_strength(self, strength: float) -> str:
        """Categorize relationship strength into levels"""
        if strength > 0.8:
            return 'very_strong'
        elif strength > 0.6:
            return 'strong'
        elif strength > 0.4:
            return 'moderate'
        elif strength > 0.2:
            return 'weak'
        else:
            return 'very_weak'
    
    def _determine_usage_recommendation(self, optimization_score: float) -> str:
        """Determine how the context should be used based on optimization score"""
        if optimization_score > 0.8:
            return 'primary_context'  # Use as primary contextual information
        elif optimization_score > 0.6:
            return 'secondary_context'  # Use as supporting context
        elif optimization_score > 0.4:
            return 'reference_only'  # Mention as reference only
        else:
            return 'exclude'  # Don't use in this context
    
    def _calculate_adaptive_threshold(self, optimized_lessons: List[Dict[str, Any]]) -> float:
        """Calculate adaptive threshold for context selection"""
        if not optimized_lessons:
            return self.optimization_config['min_quality_threshold']
        
        scores = [lesson['optimization_score'] for lesson in optimized_lessons]
        
        if not self.optimization_config['adaptive_threshold']:
            return self.optimization_config['min_quality_threshold']
        
        # Use dynamic threshold based on score distribution
        mean_score = sum(scores) / len(scores)
        max_score = max(scores)
        
        # Set threshold at 70% of the way between mean and max
        adaptive_threshold = mean_score + 0.7 * (max_score - mean_score)
        
        # Ensure it's not below minimum threshold
        return max(adaptive_threshold, self.optimization_config['min_quality_threshold'])
    
    def _assess_context_quality(self, lesson_name: str, 
                               optimized_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the quality of optimized context"""
        if not optimized_context:
            return {
                'overall_quality': 0.0,
                'context_coverage': 0.0,
                'context_diversity': 0.0,
                'context_relevance': 0.0,
                'context_balance': 0.0
            }
        
        # Coverage: How much of the topic space is covered
        coverage_score = min(len(optimized_context) / 3, 1.0)  # Optimal around 3 lessons
        
        # Diversity: How diverse are the relationship types
        relationship_types = set(
            lesson['context_factors']['relationship_type'] 
            for lesson in optimized_context
        )
        diversity_score = min(len(relationship_types) / 3, 1.0)  # Up to 3 types
        
        # Relevance: Average optimization score
        avg_opt_score = sum(
            lesson['optimization_score'] 
            for lesson in optimized_context
        ) / len(optimized_context)
        
        # Balance: How balanced are the similarity levels
        similarity_levels = [
            lesson['context_factors']['similarity_level'] 
            for lesson in optimized_context
        ]
        unique_levels = len(set(similarity_levels))
        balance_score = min(unique_levels / 3, 1.0)  # Up to 3 levels
        
        # Overall quality (weighted combination)
        overall_quality = (
            coverage_score * 0.3 +
            diversity_score * 0.25 +
            avg_opt_score * 0.3 +
            balance_score * 0.15
        )
        
        return {
            'overall_quality': overall_quality,
            'context_coverage': coverage_score,
            'context_diversity': diversity_score,
            'context_relevance': avg_opt_score,
            'context_balance': balance_score,
            'assessment_details': {
                'context_count': len(optimized_context),
                'relationship_types': list(relationship_types),
                'similarity_levels': list(set(similarity_levels))
            }
        }
    
    def _generate_optimization_recommendations(self, lesson_name: str,
                                             optimized_context: List[Dict[str, Any]],
                                             quality_assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving context optimization"""
        recommendations = []
        
        quality = quality_assessment['overall_quality']
        
        if quality < 0.3:
            recommendations.append("Context quality is low. Consider adding more related content or reviewing lesson relationships.")
        elif quality < 0.6:
            recommendations.append("Context quality is moderate. Consider enhancing lesson content or cross-references.")
        else:
            recommendations.append("Context quality is good. Current optimization is effective.")
        
        # Specific recommendations based on quality factors
        if quality_assessment['context_coverage'] < 0.5:
            recommendations.append("Consider adding more related lessons to improve coverage.")
        
        if quality_assessment['context_diversity'] < 0.5:
            recommendations.append("Diversify relationship types between lessons for better context.")
        
        if quality_assessment['context_relevance'] < 0.5:
            recommendations.append("Review similarity calculations and relationship strength assessments.")
        
        if quality_assessment['context_balance'] < 0.5:
            recommendations.append("Balance the mix of high and moderate similarity lessons.")
        
        # Usage-specific recommendations
        primary_context = [l for l in optimized_context if l['recommended_usage'] == 'primary_context']
        if not primary_context:
            recommendations.append("No primary context lessons identified. Consider strengthening lesson relationships.")
        
        return recommendations
    
    def _update_quality_history(self, lesson_name: str, quality_assessment: Dict[str, Any]) -> None:
        """Update quality history for learning and improvement"""
        if lesson_name not in self.context_quality_history:
            self.context_quality_history[lesson_name] = []
        
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'quality_score': quality_assessment['overall_quality'],
            'quality_breakdown': quality_assessment
        }
        
        self.context_quality_history[lesson_name].append(history_entry)
        
        # Keep only last 10 entries per lesson
        if len(self.context_quality_history[lesson_name]) > 10:
            self.context_quality_history[lesson_name] = self.context_quality_history[lesson_name][-10:]
    
    def optimize_all_lessons(self) -> Dict[str, Any]:
        """Optimize context for all available lessons"""
        if not self.analyzer:
            return {'error': 'Analyzer not available'}
        
        lessons = self.analyzer._get_available_lessons()
        
        if not lessons:
            print("No lessons found for optimization")
            return {}
        
        print(f"🎯 Optimizing context for {len(lessons)} lessons...")
        
        optimization_results = {}
        quality_scores = []
        
        for lesson in lessons:
            try:
                result = self.optimize_context_for_lesson(lesson)
                optimization_results[lesson] = result
                quality_scores.append(result['quality_score'])
                print(f"✓ Optimized {lesson}: quality={result['quality_score']:.3f}")
            except Exception as e:
                print(f"ERROR optimizing {lesson}: {e}")
                optimization_results[lesson] = {'error': str(e)}
        
        # Calculate overall metrics
        if quality_scores:
            overall_metrics = {
                'total_lessons': len(lessons),
                'successful_optimizations': len(quality_scores),
                'average_quality': sum(quality_scores) / len(quality_scores),
                'quality_distribution': {
                    'high': len([q for q in quality_scores if q > 0.7]),
                    'medium': len([q for q in quality_scores if 0.4 <= q <= 0.7]),
                    'low': len([q for q in quality_scores if q < 0.4])
                }
            }
        else:
            overall_metrics = {'error': 'No successful optimizations'}
        
        # Save optimization data
        self.save_optimization_data()
        
        summary = {
            'optimization_results': optimization_results,
            'overall_metrics': overall_metrics,
            'optimization_timestamp': datetime.now().isoformat()
        }
        
        print(f"📊 Optimization complete: {overall_metrics}")
        return summary
    
    def tune_optimization_parameters(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Tune optimization parameters based on feedback"""
        # This would implement machine learning-based parameter tuning
        # For now, provide a simple manual tuning interface
        
        current_config = self.optimization_config.copy()
        tuning_suggestions = []
        
        # Analyze historical quality trends
        if self.context_quality_history:
            avg_qualities = []
            for lesson_history in self.context_quality_history.values():
                if lesson_history:
                    recent_quality = lesson_history[-1]['quality_score']
                    avg_qualities.append(recent_quality)
            
            if avg_qualities:
                overall_avg = sum(avg_qualities) / len(avg_qualities)
                
                if overall_avg < 0.4:
                    tuning_suggestions.append("Consider increasing semantic_weight for better relevance")
                    tuning_suggestions.append("Consider decreasing min_quality_threshold temporarily")
                elif overall_avg > 0.8:
                    tuning_suggestions.append("Consider increasing min_quality_threshold for higher standards")
        
        return {
            'current_config': current_config,
            'tuning_suggestions': tuning_suggestions,
            'historical_performance': {
                'lessons_tracked': len(self.context_quality_history),
                'average_recent_quality': overall_avg if 'overall_avg' in locals() else None
            }
        }

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Optimize cross-lesson context selection")
    parser.add_argument("--lessons-dir", default="lessons", help="Directory containing lessons")
    parser.add_argument("--config-dir", default="config", help="Directory for configuration")
    parser.add_argument("--optimize", help="Optimize context for specific lesson")
    parser.add_argument("--optimize-all", action="store_true", help="Optimize context for all lessons")
    parser.add_argument("--quality", help="Assess quality for specific lesson")
    parser.add_argument("--tune", action="store_true", help="Show parameter tuning suggestions")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    optimizer = ContextOptimizer(args.lessons_dir, args.config_dir)
    
    if args.optimize:
        result = optimizer.optimize_context_for_lesson(args.optimize)
        print(f"\n🎯 Optimization result for {args.optimize}:")
        print(json.dumps(result, indent=2, default=str))
    
    elif args.optimize_all:
        results = optimizer.optimize_all_lessons()
        if args.verbose:
            print("\n📊 All optimization results:")
            print(json.dumps(results, indent=2, default=str))
    
    elif args.quality:
        # Get quality assessment only
        if optimizer.analyzer:
            context = optimizer.analyzer.get_context_for_lesson(args.quality)
            optimized = optimizer._apply_context_optimization(args.quality, context)
            quality = optimizer._assess_context_quality(args.quality, optimized)
            print(f"\n📈 Quality assessment for {args.quality}:")
            print(json.dumps(quality, indent=2, default=str))
        else:
            print("ERROR: Analyzer not available")
    
    elif args.tune:
        tuning_info = optimizer.tune_optimization_parameters({})
        print("\n⚙️  Parameter tuning information:")
        print(json.dumps(tuning_info, indent=2, default=str))
    
    else:
        print("Use --optimize LESSON, --optimize-all, or --help for more options")

if __name__ == "__main__":
    main()