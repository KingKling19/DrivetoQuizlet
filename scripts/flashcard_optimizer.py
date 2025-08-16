#!/usr/bin/env python3
"""
Flashcard Optimizer

Provides content balance analysis, gap detection, and optimization capabilities
for improving flashcard sets through automated and manual refinement.
Enhanced with cross-lesson context integration and performance monitoring.
"""

import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass
from pathlib import Path

# Import integration modules
try:
    from scripts.model_manager import ModelManager
    from scripts.performance_monitor import performance_monitor
    from scripts.cross_lesson_analyzer import CrossLessonAnalyzer
except ImportError as e:
    print(f"Warning: Could not import integration modules: {e}")
    model_manager = None
    performance_monitor = None
    CrossLessonAnalyzer = None

@dataclass
class OptimizationResult:
    """Result of flashcard optimization"""
    original_count: int
    optimized_count: int
    improvements_made: List[str]
    quality_improvement: float
    content_balance_improvement: float
    optimization_time: float
    cross_lesson_context_used: bool
    performance_metrics: Dict[str, Any]

class FlashcardOptimizer:
    """Main optimizer for flashcard sets with enhanced integration"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the optimizer with configuration"""
        self.config = config or self._load_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize integration components
        self.cross_lesson_analyzer = None
        self.performance_tracker = None
        
        if CrossLessonAnalyzer:
            try:
                self.cross_lesson_analyzer = CrossLessonAnalyzer()
            except Exception as e:
                self.logger.warning(f"Could not initialize cross-lesson analyzer: {e}")
        
        if performance_monitor:
            self.performance_tracker = performance_monitor
        
        # Content balance targets
        self.content_targets = self.config.get('content_balance', {}).get('target_distribution', {
            'operations': 0.25,
            'communications': 0.15,
            'equipment': 0.20,
            'procedures': 0.20,
            'safety': 0.10,
            'leadership': 0.10
        })
        
        # Quality thresholds
        self.quality_thresholds = self.config.get('quality_thresholds', {
            'min_quality_score': 0.6,
            'min_definition_length': 50,
            'max_definition_length': 600,
            'duplicate_similarity_threshold': 0.85
        })
        
        # Integration settings
        self.integration_config = self.config.get('integration', {})
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from file"""
        try:
            config_path = Path("config/flashcard_optimization_config.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load config file: {e}")
        
        return {}
    
    def optimize_flashcards(self, flashcards: List[Dict[str, Any]], lesson_id: Optional[str] = None) -> OptimizationResult:
        """Main optimization function for flashcard sets with enhanced integration"""
        start_time = time.time()
        
        try:
            original_count = len(flashcards)
            optimized = flashcards.copy()
            
            improvements = []
            quality_improvement = 0.0
            balance_improvement = 0.0
            cross_lesson_context_used = False
            
            # Track performance if enabled
            if self.performance_tracker and self.integration_config.get('performance_monitoring', {}).get('enabled', False):
                self.performance_tracker.start_operation('flashcard_optimization', {
                    'lesson_id': lesson_id,
                    'original_count': original_count
                })
            
            # Step 1: Enhance with cross-lesson context
            if (self.cross_lesson_analyzer and 
                self.integration_config.get('cross_lesson_context', {}).get('enabled', False) and 
                lesson_id):
                
                try:
                    optimized, context_enhancements = self._enhance_with_cross_lesson_context(optimized, lesson_id)
                    if context_enhancements:
                        improvements.extend(context_enhancements)
                        cross_lesson_context_used = True
                        self.logger.info(f"Enhanced {len(optimized)} flashcards with cross-lesson context")
                except Exception as e:
                    self.logger.warning(f"Cross-lesson context enhancement failed: {e}")
            
            # Step 2: Remove duplicates
            optimized, dupes_removed = self._remove_duplicates(optimized)
            if dupes_removed > 0:
                improvements.append(f"Removed {dupes_removed} duplicate flashcards")
            
            # Step 3: Quality filtering
            optimized, quality_filtered = self._filter_low_quality(optimized)
            if quality_filtered > 0:
                improvements.append(f"Filtered out {quality_filtered} low-quality flashcards")
            
            # Step 4: Content balance optimization
            optimized, balance_changes = self._optimize_content_balance(optimized)
            if balance_changes:
                improvements.extend(balance_changes)
            
            # Step 5: Quality improvement
            optimized, quality_improvements = self._improve_quality(optimized)
            if quality_improvements:
                improvements.extend(quality_improvements)
            
            # Calculate improvements
            if original_count > 0:
                quality_improvement = self._calculate_quality_improvement(flashcards, optimized)
                balance_improvement = self._calculate_balance_improvement(optimized)
            
            optimization_time = time.time() - start_time
            
            # Log performance metrics
            performance_metrics = {}
            if self.performance_tracker and self.integration_config.get('performance_monitoring', {}).get('enabled', False):
                performance_metrics = {
                    'optimization_time': optimization_time,
                    'quality_improvement': quality_improvement,
                    'content_balance_improvement': balance_improvement,
                    'cross_lesson_context_used': cross_lesson_context_used
                }
                self.performance_tracker.end_operation('flashcard_optimization', performance_metrics)
                
                # Log optimization events if enabled
                if self.integration_config.get('performance_monitoring', {}).get('log_optimization_events', False):
                    self.performance_tracker.log_event('flashcard_optimization_completed', {
                        'lesson_id': lesson_id,
                        'original_count': original_count,
                        'optimized_count': len(optimized),
                        'improvements': improvements,
                        'performance_metrics': performance_metrics
                    })
            
            self.logger.info(f"Optimization complete: {len(optimized)} flashcards remaining in {optimization_time:.2f}s")
            
            return OptimizationResult(
                original_count=original_count,
                optimized_count=len(optimized),
                improvements_made=improvements,
                quality_improvement=quality_improvement,
                content_balance_improvement=balance_improvement,
                optimization_time=optimization_time,
                cross_lesson_context_used=cross_lesson_context_used,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error optimizing flashcards: {e}")
            optimization_time = time.time() - start_time
            
            # Log error in performance monitor
            if self.performance_tracker:
                self.performance_tracker.log_event('flashcard_optimization_error', {
                    'lesson_id': lesson_id,
                    'error': str(e),
                    'optimization_time': optimization_time
                })
            
            return OptimizationResult(
                original_count=len(flashcards),
                optimized_count=len(flashcards),
                improvements_made=[f"Error during optimization: {e}"],
                quality_improvement=0.0,
                content_balance_improvement=0.0,
                optimization_time=optimization_time,
                cross_lesson_context_used=False,
                performance_metrics={'error': str(e)}
            )
    
    def _enhance_with_cross_lesson_context(self, flashcards: List[Dict[str, Any]], lesson_id: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Enhance flashcards with cross-lesson context"""
        enhancements = []
        
        try:
            # Get related lessons
            related_lessons = self.cross_lesson_analyzer.get_related_lessons(
                lesson_id, 
                max_lessons=self.integration_config.get('cross_lesson_context', {}).get('max_related_lessons', 5)
            )
            
            if not related_lessons:
                return flashcards, enhancements
            
            # Get context from related lessons
            context_data = self.cross_lesson_analyzer.get_lesson_context(related_lessons)
            
            # Enhance flashcards with context
            enhanced_count = 0
            for flashcard in flashcards:
                enhanced = self._enhance_single_flashcard_with_context(flashcard, context_data)
                if enhanced:
                    enhanced_count += 1
            
            if enhanced_count > 0:
                enhancements.append(f"Enhanced {enhanced_count} flashcards with cross-lesson context from {len(related_lessons)} related lessons")
            
            return flashcards, enhancements
            
        except Exception as e:
            self.logger.warning(f"Cross-lesson context enhancement failed: {e}")
            return flashcards, enhancements
    
    def _enhance_single_flashcard_with_context(self, flashcard: Dict[str, Any], context_data: Dict[str, Any]) -> bool:
        """Enhance a single flashcard with cross-lesson context"""
        try:
            term = flashcard.get('term', '').lower()
            definition = flashcard.get('definition', '')
            
            # Find relevant context
            relevant_context = []
            for lesson_id, lesson_context in context_data.items():
                if any(keyword in term or keyword in definition.lower() 
                      for keyword in lesson_context.get('keywords', [])):
                    relevant_context.append(lesson_context)
            
            if relevant_context:
                # Enhance definition with context
                context_weight = self.integration_config.get('cross_lesson_context', {}).get('context_weight', 0.15)
                
                # Add context information to flashcard metadata
                if 'metadata' not in flashcard:
                    flashcard['metadata'] = {}
                
                flashcard['metadata']['cross_lesson_context'] = {
                    'related_lessons': len(relevant_context),
                    'context_weight': context_weight,
                    'enhanced': True
                }
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Error enhancing flashcard with context: {e}")
            return False
    
    def analyze_content_distribution(self, flashcards: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze content distribution across topics"""
        try:
            if not flashcards:
                return {'error': 'No flashcards provided'}
            
            # Count flashcards by topic
            topic_counts = defaultdict(int)
            topic_flashcards = defaultdict(list)
            
            for flashcard in flashcards:
                topic = self._extract_topic(flashcard)
                topic_counts[topic] += 1
                topic_flashcards[topic].append(flashcard)
            
            total_flashcards = len(flashcards)
            
            # Calculate distribution percentages
            distribution = {}
            for topic, count in topic_counts.items():
                percentage = (count / total_flashcards) * 100
                target_percentage = self.content_targets.get(topic, 0) * 100
                
                distribution[topic] = {
                    'count': count,
                    'percentage': round(percentage, 1),
                    'target_percentage': round(target_percentage, 1),
                    'deviation': round(percentage - target_percentage, 1),
                    'flashcards': topic_flashcards[topic]
                }
            
            # Calculate overall balance score
            balance_score = self._calculate_balance_score(distribution)
            
            return {
                'total_flashcards': total_flashcards,
                'distribution': distribution,
                'balance_score': round(balance_score, 3),
                'balance_level': self._get_balance_level(balance_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing content distribution: {e}")
            return {'error': str(e)}
    
    def identify_content_gaps(self, flashcards: List[Dict[str, Any]], lesson_content: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Identify content gaps in flashcard coverage"""
        try:
            gaps = []
            
            # Analyze current distribution
            distribution = self.analyze_content_distribution(flashcards)
            if 'error' in distribution:
                return gaps
            
            # Check each topic for gaps
            for topic, data in distribution['distribution'].items():
                target_percentage = self.content_targets.get(topic, 0) * 100
                current_percentage = data['percentage']
                
                # Identify significant gaps (more than 10% below target)
                if current_percentage < (target_percentage - 10):
                    gap_size = target_percentage - current_percentage
                    estimated_cards_needed = int((gap_size / 100) * len(flashcards))
                    
                    gaps.append({
                        'topic': topic,
                        'current_coverage': current_percentage,
                        'target_coverage': target_percentage,
                        'gap_size': round(gap_size, 1),
                        'estimated_cards_needed': max(1, estimated_cards_needed),
                        'priority': 'high' if gap_size > 20 else 'medium',
                        'suggested_terms': self._suggest_terms_for_topic(topic, lesson_content)
                    })
            
            # Sort by priority and gap size
            gaps.sort(key=lambda x: (x['priority'] == 'high', -x['gap_size']), reverse=True)
            
            return gaps
            
        except Exception as e:
            self.logger.error(f"Error identifying content gaps: {e}")
            return []
    
    def balance_content_coverage(self, flashcards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Balance content coverage by topic"""
        try:
            balanced = flashcards.copy()
            
            # Analyze current distribution
            distribution = self.analyze_content_distribution(flashcards)
            if 'error' in distribution:
                return balanced
            
            # Identify topics that are over-represented
            over_represented = []
            under_represented = []
            
            for topic, data in distribution['distribution'].items():
                target_percentage = self.content_targets.get(topic, 0) * 100
                current_percentage = data['percentage']
                
                if current_percentage > (target_percentage + 10):
                    over_represented.append({
                        'topic': topic,
                        'excess': current_percentage - target_percentage,
                        'flashcards': data['flashcards']
                    })
                elif current_percentage < (target_percentage - 10):
                    under_represented.append({
                        'topic': topic,
                        'shortage': target_percentage - current_percentage
                    })
            
            # Prioritize flashcards from over-represented topics
            for over_topic in over_represented:
                topic_flashcards = over_topic['flashcards']
                # Sort by quality score (keep the best ones)
                topic_flashcards.sort(key=lambda x: float(x.get('quality_score', 0)), reverse=True)
                
                # Calculate how many to keep
                target_count = int((self.content_targets.get(over_topic['topic'], 0) * len(flashcards)))
                keep_count = min(len(topic_flashcards), target_count)
                
                # Remove excess flashcards
                excess_flashcards = topic_flashcards[keep_count:]
                for fc in excess_flashcards:
                    if fc in balanced:
                        balanced.remove(fc)
            
            self.logger.info(f"Content balance optimization complete: {len(balanced)} flashcards remaining")
            return balanced
            
        except Exception as e:
            self.logger.error(f"Error balancing content coverage: {e}")
            return flashcards
    
    def assess_topic_coverage(self, flashcards: List[Dict[str, Any]], lesson_topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Assess topic coverage against lesson content"""
        try:
            if not flashcards:
                return {'error': 'No flashcards provided'}
            
            # Extract topics from flashcards
            flashcard_topics = set()
            for flashcard in flashcards:
                topic = self._extract_topic(flashcard)
                flashcard_topics.add(topic)
            
            # Compare with lesson topics if provided
            coverage_analysis = {
                'flashcard_topics': list(flashcard_topics),
                'total_unique_topics': len(flashcard_topics),
                'topic_distribution': self._get_topic_distribution(flashcards)
            }
            
            if lesson_topics:
                lesson_topic_set = set(lesson_topics)
                covered_topics = flashcard_topics.intersection(lesson_topic_set)
                missing_topics = lesson_topic_set - flashcard_topics
                
                coverage_analysis.update({
                    'lesson_topics': list(lesson_topic_set),
                    'covered_topics': list(covered_topics),
                    'missing_topics': list(missing_topics),
                    'coverage_percentage': round((len(covered_topics) / len(lesson_topic_set)) * 100, 1) if lesson_topic_set else 0
                })
            
            return coverage_analysis
            
        except Exception as e:
            self.logger.error(f"Error assessing topic coverage: {e}")
            return {'error': str(e)}
    
    def is_duplicate_flashcard(self, card1: Dict[str, Any], card2: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> bool:
        """Check if two flashcards are duplicates"""
        try:
            if config is None:
                config = self.quality_thresholds
            
            term1 = card1.get('term', '').strip().lower()
            term2 = card2.get('term', '').strip().lower()
            def1 = card1.get('definition', '').strip().lower()
            def2 = card2.get('definition', '').strip().lower()
            
            # Exact term match
            if term1 == term2:
                return True
            
            # Fuzzy term similarity
            if self._calculate_term_similarity(term1, term2) > 0.8:
                # Check definition similarity
                if self._calculate_definition_similarity(def1, def2) > config.get('duplicate_similarity_threshold', 0.85):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking for duplicates: {e}")
            return False
    
    def _remove_duplicates(self, flashcards: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """Remove duplicate flashcards"""
        try:
            unique_flashcards = []
            duplicates_removed = 0
            
            for flashcard in flashcards:
                is_duplicate = False
                
                for existing in unique_flashcards:
                    if self.is_duplicate_flashcard(flashcard, existing):
                        is_duplicate = True
                        # Keep the one with higher quality score
                        if float(flashcard.get('quality_score', 0)) > float(existing.get('quality_score', 0)):
                            unique_flashcards.remove(existing)
                            unique_flashcards.append(flashcard)
                        break
                
                if not is_duplicate:
                    unique_flashcards.append(flashcard)
                else:
                    duplicates_removed += 1
            
            return unique_flashcards, duplicates_removed
            
        except Exception as e:
            self.logger.error(f"Error removing duplicates: {e}")
            return flashcards, 0
    
    def _filter_low_quality(self, flashcards: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """Filter out low-quality flashcards"""
        try:
            min_quality = self.quality_thresholds.get('min_quality_score', 0.6)
            filtered = []
            removed_count = 0
            
            for flashcard in flashcards:
                quality_score = float(flashcard.get('quality_score', 0))
                if quality_score >= min_quality:
                    filtered.append(flashcard)
                else:
                    removed_count += 1
            
            return filtered, removed_count
            
        except Exception as e:
            self.logger.error(f"Error filtering low quality: {e}")
            return flashcards, 0
    
    def _optimize_content_balance(self, flashcards: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Optimize content balance"""
        try:
            changes = []
            balanced = self.balance_content_coverage(flashcards)
            
            if len(balanced) != len(flashcards):
                changes.append(f"Content balance optimization: {len(flashcards)} -> {len(balanced)} flashcards")
            
            return balanced, changes
            
        except Exception as e:
            self.logger.error(f"Error optimizing content balance: {e}")
            return flashcards, []
    
    def _improve_quality(self, flashcards: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Improve flashcard quality"""
        try:
            improvements = []
            improved_flashcards = []
            
            for flashcard in flashcards:
                improved = flashcard.copy()
                
                # Improve definition length if needed
                definition = flashcard.get('definition', '')
                if len(definition) < self.quality_thresholds.get('min_definition_length', 50):
                    # Add placeholder for improvement
                    improved['definition'] = f"{definition} (Enhanced with additional context)"
                    improvements.append("Enhanced short definitions")
                
                # Improve term clarity if needed
                term = flashcard.get('term', '')
                if len(term.split()) > 5:  # Too long term
                    # Suggest shorter version
                    words = term.split()
                    improved['term'] = ' '.join(words[:3])  # Keep first 3 words
                    improvements.append("Shortened overly long terms")
                
                improved_flashcards.append(improved)
            
            return improved_flashcards, list(set(improvements))
            
        except Exception as e:
            self.logger.error(f"Error improving quality: {e}")
            return flashcards, []
    
    def _extract_topic(self, flashcard: Dict[str, Any]) -> str:
        """Extract topic from flashcard"""
        term = flashcard.get('term', '').lower()
        definition = flashcard.get('definition', '').lower()
        
        text = f"{term} {definition}"
        
        # Topic keywords mapping
        topic_keywords = {
            'operations': ['operation', 'operational', 'mission', 'tactical', 'strategic', 'deployment'],
            'communications': ['communication', 'radio', 'signal', 'transmission', 'frequency'],
            'equipment': ['equipment', 'system', 'device', 'tool', 'gear', 'weapon'],
            'procedures': ['procedure', 'protocol', 'process', 'method', 'technique'],
            'safety': ['safety', 'security', 'protection', 'hazard', 'risk'],
            'leadership': ['leadership', 'command', 'supervision', 'management', 'authority']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text for keyword in keywords):
                return topic
        
        return 'general'
    
    def _calculate_term_similarity(self, term1: str, term2: str) -> float:
        """Calculate similarity between terms"""
        if not term1 or not term2:
            return 0.0
        
        # Simple word overlap similarity
        words1 = set(term1.split())
        words2 = set(term2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_definition_similarity(self, def1: str, def2: str) -> float:
        """Calculate similarity between definitions"""
        if not def1 or not def2:
            return 0.0
        
        # Simple word overlap similarity
        words1 = set(re.findall(r'\b\w+\b', def1.lower()))
        words2 = set(re.findall(r'\b\w+\b', def2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_balance_score(self, distribution: Dict[str, Any]) -> float:
        """Calculate overall balance score"""
        try:
            total_deviation = 0
            topic_count = 0
            
            for topic_data in distribution.values():
                deviation = abs(topic_data.get('deviation', 0))
                total_deviation += deviation
                topic_count += 1
            
            if topic_count == 0:
                return 0.0
            
            avg_deviation = total_deviation / topic_count
            # Convert to score (0 = perfect balance, 1 = poor balance)
            balance_score = min(1.0, avg_deviation / 50)  # Normalize to 0-1 scale
            
            return 1.0 - balance_score  # Invert so higher is better
            
        except Exception as e:
            self.logger.error(f"Error calculating balance score: {e}")
            return 0.0
    
    def _get_balance_level(self, balance_score: float) -> str:
        """Get balance level description"""
        if balance_score >= 0.8:
            return 'excellent'
        elif balance_score >= 0.6:
            return 'good'
        elif balance_score >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _get_topic_distribution(self, flashcards: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get topic distribution for flashcards"""
        distribution = defaultdict(int)
        for flashcard in flashcards:
            topic = self._extract_topic(flashcard)
            distribution[topic] += 1
        return dict(distribution)
    
    def _suggest_terms_for_topic(self, topic: str, lesson_content: Optional[Dict[str, Any]] = None) -> List[str]:
        """Suggest terms for a specific topic"""
        # Placeholder for term suggestions
        suggestions = {
            'operations': ['tactical planning', 'mission execution', 'operational readiness'],
            'communications': ['radio protocol', 'signal transmission', 'network security'],
            'equipment': ['system maintenance', 'device operation', 'gear inspection'],
            'procedures': ['standard operating procedure', 'protocol compliance', 'method validation'],
            'safety': ['hazard assessment', 'risk mitigation', 'safety protocol'],
            'leadership': ['command structure', 'supervision techniques', 'management principles']
        }
        
        return suggestions.get(topic, [])
    
    def _calculate_quality_improvement(self, original: List[Dict[str, Any]], optimized: List[Dict[str, Any]]) -> float:
        """Calculate quality improvement percentage"""
        try:
            if not original or not optimized:
                return 0.0
            
            original_avg = sum(float(fc.get('quality_score', 0)) for fc in original) / len(original)
            optimized_avg = sum(float(fc.get('quality_score', 0)) for fc in optimized) / len(optimized)
            
            if original_avg == 0:
                return 0.0
            
            improvement = (optimized_avg - original_avg) / original_avg
            return max(0.0, improvement)
            
        except Exception as e:
            self.logger.error(f"Error calculating quality improvement: {e}")
            return 0.0
    
    def _calculate_balance_improvement(self, optimized: List[Dict[str, Any]]) -> float:
        """Calculate balance improvement"""
        try:
            distribution = self.analyze_content_distribution(optimized)
            if 'error' in distribution:
                return 0.0
            
            return distribution.get('balance_score', 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating balance improvement: {e}")
            return 0.0
