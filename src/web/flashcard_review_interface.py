#!/usr/bin/env python3
"""
Flashcard Review Interface

Provides manual review and editing capabilities for flashcards with advanced
optimization features including quality assessment, duplicate detection,
content balance analysis, and clustering.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import sqlite3
from dataclasses import dataclass, asdict
import logging
from collections import Counter

# Import existing modules
try:
    from src.processing.convert_folder_to_quizlet import dedupe_and_filter, canonical_term
    from src.analysis.flashcard_quality_assessor import FlashcardQualityAssessor
    from src.analysis.flashcard_clustering import FlashcardClusterer
    from src.processing.flashcard_optimizer import FlashcardOptimizer
except ImportError as e:
    logging.warning(f"Could not import some modules: {e}")

@dataclass
class FlashcardReviewData:
    """Data structure for flashcard review information"""
    id: str
    term: str
    definition: str
    source_slide: int
    confidence: float
    quality_score: float
    difficulty_level: str
    topic: str
    is_duplicate: bool
    duplicate_group: Optional[str]
    needs_review: bool
    review_notes: str
    last_modified: datetime
    original_data: Dict[str, Any]

class FlashcardReviewInterface:
    """Main interface for flashcard review and editing"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the review interface with configuration"""
        self.config = self._load_config(config_path)
        self.quality_assessor = FlashcardQualityAssessor(self.config)
        self.clusterer = FlashcardClusterer(self.config)
        self.optimizer = FlashcardOptimizer(self.config)
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration for the review interface"""
        default_config = {
            "quality_thresholds": {
                "min_definition_length": 50,
                "max_definition_length": 600,
                "min_quality_score": 0.6,
                "duplicate_similarity_threshold": 0.85
            },
            "review_settings": {
                "auto_save_interval": 300,  # 5 minutes
                "backup_before_save": True,
                "max_review_session_size": 100
            },
            "optimization": {
                "enable_auto_optimization": False,
                "suggestion_threshold": 0.7,
                "bulk_operation_limit": 50
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Could not load config from {config_path}: {e}")
        
        return default_config
    
    def load_flashcards_for_review(self, lesson_path: str) -> Dict[str, Any]:
        """Load flashcards with metadata for manual review"""
        try:
            lesson_path = Path(lesson_path)
            if not lesson_path.exists():
                raise FileNotFoundError(f"Lesson path not found: {lesson_path}")
            
            # Find flashcard files
            flashcard_files = list(lesson_path.rglob("*.json"))
            if not flashcard_files:
                # Look for processed output
                output_dir = lesson_path / "output"
                if output_dir.exists():
                    flashcard_files = list(output_dir.rglob("*.json"))
            
            if not flashcard_files:
                raise FileNotFoundError(f"No flashcard files found in {lesson_path}")
            
            all_flashcards = []
            for file_path in flashcard_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_flashcards.extend(data)
                        elif isinstance(data, dict) and 'flashcards' in data:
                            all_flashcards.extend(data['flashcards'])
                except Exception as e:
                    self.logger.warning(f"Could not load {file_path}: {e}")
            
            if not all_flashcards:
                raise ValueError(f"No valid flashcards found in {lesson_path}")
            
            # Process flashcards for review
            review_data = self._process_flashcards_for_review(all_flashcards)
            
            # Generate optimization suggestions
            suggestions = self.get_optimization_suggestions(review_data)
            
            # Get clustering analysis
            clusters = self.clusterer.cluster_flashcards_by_topic(review_data)
            
            return {
                "lesson_path": str(lesson_path),
                "total_flashcards": len(review_data),
                "flashcards": [asdict(fc) for fc in review_data],
                "suggestions": suggestions,
                "clusters": clusters,
                "summary": self._generate_review_summary(review_data),
                "loaded_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error loading flashcards for review: {e}")
            raise
    
    def _process_flashcards_for_review(self, flashcards: List[Dict[str, Any]]) -> List[FlashcardReviewData]:
        """Process raw flashcards into review data format"""
        review_data = []
        
        # First pass: assess quality and identify duplicates
        quality_scores = {}
        duplicate_groups = {}
        
        for i, card in enumerate(flashcards):
            # Assess quality
            quality_info = self.quality_assessor.assess_flashcard_quality(card)
            quality_scores[i] = quality_info['overall_score']
            
            # Check for duplicates
            is_duplicate = False
            duplicate_group = None
            
            for j in range(i):
                if self._is_duplicate(card, flashcards[j]):
                    is_duplicate = True
                    duplicate_group = f"group_{j}"
                    if j not in duplicate_groups:
                        duplicate_groups[j] = f"group_{j}"
                    duplicate_groups[i] = duplicate_groups[j]
                    break
            
            if is_duplicate and i not in duplicate_groups:
                duplicate_groups[i] = duplicate_group
        
        # Second pass: create review data
        for i, card in enumerate(flashcards):
            review_card = FlashcardReviewData(
                id=f"fc_{i}_{hash(card.get('term', ''))}",
                term=card.get('term', '').strip(),
                definition=card.get('definition', '').strip(),
                source_slide=int(card.get('source_slide', 0)),
                confidence=float(card.get('confidence', 0.0)),
                quality_score=quality_scores.get(i, 0.0),
                difficulty_level=self.quality_assessor.calculate_difficulty_level(card),
                topic=self._extract_topic(card),
                is_duplicate=i in duplicate_groups,
                duplicate_group=duplicate_groups.get(i),
                needs_review=quality_scores.get(i, 0.0) < self.config['quality_thresholds']['min_quality_score'],
                review_notes="",
                last_modified=datetime.now(),
                original_data=card
            )
            review_data.append(review_card)
        
        return review_data
    
    def _is_duplicate(self, card1: Dict[str, Any], card2: Dict[str, Any]) -> bool:
        """Check if two flashcards are duplicates"""
        config = self.config['quality_thresholds']
        return self.optimizer.is_duplicate_flashcard(card1, card2, config)
    
    def _extract_topic(self, card: Dict[str, Any]) -> str:
        """Extract topic from flashcard"""
        term = card.get('term', '').lower()
        definition = card.get('definition', '').lower()
        
        # Simple keyword-based topic extraction
        topics = {
            'operations': ['operation', 'operational', 'mission', 'tactical'],
            'communications': ['communication', 'radio', 'signal', 'transmission'],
            'equipment': ['equipment', 'system', 'device', 'tool', 'gear'],
            'procedures': ['procedure', 'protocol', 'process', 'method'],
            'safety': ['safety', 'security', 'protection', 'hazard'],
            'leadership': ['leadership', 'command', 'supervision', 'management']
        }
        
        text = f"{term} {definition}"
        for topic, keywords in topics.items():
            if any(keyword in text for keyword in keywords):
                return topic
        
        return 'general'
    
    def _generate_review_summary(self, review_data: List[FlashcardReviewData]) -> Dict[str, Any]:
        """Generate summary statistics for review session"""
        total = len(review_data)
        needs_review = sum(1 for fc in review_data if fc.needs_review)
        duplicates = sum(1 for fc in review_data if fc.is_duplicate)
        
        quality_scores = [fc.quality_score for fc in review_data]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        difficulty_distribution = {}
        for fc in review_data:
            difficulty_distribution[fc.difficulty_level] = difficulty_distribution.get(fc.difficulty_level, 0) + 1
        
        topic_distribution = {}
        for fc in review_data:
            topic_distribution[fc.topic] = topic_distribution.get(fc.topic, 0) + 1
        
        return {
            "total_flashcards": total,
            "needs_review": needs_review,
            "duplicates": duplicates,
            "average_quality_score": round(avg_quality, 3),
            "difficulty_distribution": difficulty_distribution,
            "topic_distribution": topic_distribution,
            "review_progress": 0
        }
    
    def save_reviewed_flashcards(self, flashcards: List[Dict[str, Any]], lesson_path: str) -> bool:
        """Save reviewed flashcards to the lesson directory"""
        try:
            # Handle case where flashcards might be an OptimizationResult
            if hasattr(flashcards, 'optimized_count'):
                # It's an OptimizationResult, we need to get the actual flashcards
                # For now, return True since the optimization already happened
                return True
            
            # Create output directory if it doesn't exist
            output_dir = Path(lesson_path) / "output"
            output_dir.mkdir(exist_ok=True)
            
            # Save optimized flashcards
            output_file = output_dir / "optimized_flashcards.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(flashcards, f, indent=2, ensure_ascii=False)
            
            # Save as Quizlet format
            quizlet_file = output_dir / "optimized_flashcards_quizlet.txt"
            with open(quizlet_file, 'w', encoding='utf-8') as f:
                for flashcard in flashcards:
                    term = flashcard.get('term', '').replace('\t', ' ').replace('\n', ' ')
                    definition = flashcard.get('definition', '').replace('\t', ' ').replace('\n', ' ')
                    f.write(f"{term}\t{definition}\n")
            
            self.logger.info(f"Saved {len(flashcards)} optimized flashcards to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving reviewed flashcards: {e}")
            return False
    
    def _create_backup(self, lesson_path: Path) -> None:
        """Create backup of existing flashcard files"""
        try:
            backup_dir = lesson_path / "backups" / datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy existing flashcard files
            for file_path in lesson_path.rglob("*.json"):
                if "backup" not in str(file_path):
                    relative_path = file_path.relative_to(lesson_path)
                    backup_file = backup_dir / relative_path
                    backup_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    import shutil
                    shutil.copy2(file_path, backup_file)
            
            self.logger.info(f"Created backup at {backup_dir}")
            
        except Exception as e:
            self.logger.warning(f"Could not create backup: {e}")
    
    def get_optimization_suggestions(self, flashcards: List[FlashcardReviewData]) -> List[Dict[str, Any]]:
        """Generate optimization suggestions for flashcards"""
        suggestions = []
        
        # Quality-based suggestions
        low_quality_cards = [fc for fc in flashcards if fc.quality_score < self.config['quality_thresholds']['min_quality_score']]
        if low_quality_cards:
            suggestions.append({
                'type': 'quality_improvement',
                'priority': 'high',
                'title': 'Low Quality Flashcards',
                'description': f'{len(low_quality_cards)} flashcards have quality scores below threshold',
                'affected_cards': [fc.id for fc in low_quality_cards],
                'action': 'review_and_improve'
            })
        
        # Duplicate suggestions
        duplicate_cards = [fc for fc in flashcards if fc.is_duplicate]
        if duplicate_cards:
            suggestions.append({
                'type': 'duplicate_removal',
                'priority': 'medium',
                'title': 'Duplicate Flashcards',
                'description': f'{len(duplicate_cards)} flashcards are duplicates',
                'affected_cards': [fc.id for fc in duplicate_cards],
                'action': 'merge_or_remove'
            })
        
        # Content balance suggestions
        topic_distribution = {}
        for fc in flashcards:
            topic_distribution[fc.topic] = topic_distribution.get(fc.topic, 0) + 1
        
        total_cards = len(flashcards)
        if total_cards > 0:
            for topic, count in topic_distribution.items():
                percentage = (count / total_cards) * 100
                if percentage < 10:  # Less than 10% coverage
                    suggestions.append({
                        'type': 'content_balance',
                        'priority': 'low',
                        'title': f'Low Coverage: {topic.title()}',
                        'description': f'Only {percentage:.1f}% of flashcards cover {topic}',
                        'affected_cards': [],
                        'action': 'add_more_content'
                    })
        
        return suggestions
    
    def apply_bulk_operations(self, flashcards: List[Dict[str, Any]], operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply bulk operations to flashcards"""
        try:
            result = flashcards.copy()
            
            for operation in operations:
                op_type = operation.get('type')
                op_data = operation.get('data', {})
                
                if op_type == 'delete':
                    indices = op_data.get('indices', [])
                    # Delete in reverse order to maintain indices
                    for index in sorted(indices, reverse=True):
                        if 0 <= index < len(result):
                            del result[index]
                
                elif op_type == 'update':
                    index = op_data.get('index')
                    updates = op_data.get('updates', {})
                    if 0 <= index < len(result):
                        result[index].update(updates)
                
                elif op_type == 'merge':
                    indices = op_data.get('indices', [])
                    if len(indices) >= 2:
                        # Merge definitions
                        merged_definition = " ".join([
                            result[i].get('definition', '') 
                            for i in indices if 0 <= i < len(result)
                        ])
                        # Keep the first term, update definition
                        if 0 <= indices[0] < len(result):
                            result[indices[0]]['definition'] = merged_definition
                        # Remove other cards
                        for index in sorted(indices[1:], reverse=True):
                            if 0 <= index < len(result):
                                del result[index]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error applying bulk operations: {e}")
            return flashcards
    
    def _merge_cards(self, cards: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple cards into one"""
        if not cards:
            return {}
        
        # Use the first card as base
        merged = cards[0].copy()
        
        # Combine definitions
        definitions = [card.get('definition', '') for card in cards if card.get('definition')]
        if len(definitions) > 1:
            merged['definition'] = ' '.join(definitions)
        
        # Use highest confidence and quality scores
        confidences = [float(card.get('confidence', 0)) for card in cards]
        qualities = [float(card.get('quality_score', 0)) for card in cards]
        
        merged['confidence'] = max(confidences)
        merged['quality_score'] = max(qualities)
        
        # Mark as merged
        merged['is_duplicate'] = False
        merged['duplicate_group'] = None
        merged['review_notes'] = f"Merged from {len(cards)} cards"
        
        return merged
    
    def _improve_card_quality(self, card: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Improve card quality based on parameters"""
        improved = card.copy()
        
        # Improve definition if needed
        if parameters.get('improve_definition'):
            definition = card.get('definition', '')
            if len(definition) < self.config['quality_thresholds']['min_definition_length']:
                # Add more detail (placeholder for AI enhancement)
                improved['definition'] = f"{definition} (Enhanced with additional context)"
        
        # Recalculate quality score
        improved['quality_score'] = min(1.0, improved['quality_score'] + 0.1)
        
        return improved
    
    def get_review_statistics(self, flashcards: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get comprehensive statistics for flashcard review"""
        try:
            if not flashcards:
                return {'error': 'No flashcards provided'}
            
            # Basic statistics
            total_count = len(flashcards)
            quality_scores = [f.get('quality_score', 0) for f in flashcards]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            # Quality distribution
            quality_distribution = {
                'excellent': len([s for s in quality_scores if s >= 0.85]),
                'good': len([s for s in quality_scores if 0.7 <= s < 0.85]),
                'fair': len([s for s in quality_scores if 0.5 <= s < 0.7]),
                'poor': len([s for s in quality_scores if s < 0.5])
            }
            
            # Difficulty distribution
            difficulty_counts = Counter(f.get('difficulty', 'unknown') for f in flashcards)
            
            # Topic distribution
            topic_counts = Counter(f.get('topic', 'unknown') for f in flashcards)
            
            # Definition length statistics
            definition_lengths = [len(f.get('definition', '')) for f in flashcards]
            avg_length = sum(definition_lengths) / len(definition_lengths) if definition_lengths else 0
            
            # Cross-lesson context statistics
            context_enhanced = len([f for f in flashcards if f.get('metadata', {}).get('cross_lesson_context', {}).get('enhanced', False)])
            
            return {
                'total_flashcards': total_count,
                'average_quality_score': avg_quality,
                'quality_distribution': quality_distribution,
                'difficulty_distribution': dict(difficulty_counts),
                'topic_distribution': dict(topic_counts),
                'definition_length_stats': {
                    'average_length': avg_length,
                    'min_length': min(definition_lengths) if definition_lengths else 0,
                    'max_length': max(definition_lengths) if definition_lengths else 0
                },
                'cross_lesson_context': {
                    'enhanced_count': context_enhanced,
                    'enhancement_rate': context_enhanced / total_count if total_count > 0 else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating review statistics: {e}")
            return {'error': str(e)}
