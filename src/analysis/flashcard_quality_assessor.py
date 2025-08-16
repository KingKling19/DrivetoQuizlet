#!/usr/bin/env python3
"""
Flashcard Quality Assessor

Provides comprehensive quality assessment for flashcards including definition clarity,
difficulty assessment, military context validation, and overall quality scoring.
"""

import re
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

@dataclass
class QualityMetrics:
    """Quality metrics for a flashcard"""
    definition_length_score: float
    term_complexity_score: float
    definition_clarity_score: float
    military_relevance_score: float
    testability_score: float
    completeness_score: float
    overall_score: float

class FlashcardQualityAssessor:
    """Assesses flashcard quality using multiple metrics"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the quality assessor with configuration"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Military-specific keywords for relevance scoring
        self.military_keywords = {
            'operations': ['operation', 'mission', 'tactical', 'strategic', 'deployment'],
            'communications': ['radio', 'signal', 'communication', 'transmission', 'frequency'],
            'equipment': ['equipment', 'system', 'device', 'gear', 'tool', 'weapon'],
            'procedures': ['procedure', 'protocol', 'process', 'method', 'technique'],
            'safety': ['safety', 'security', 'protection', 'hazard', 'risk'],
            'leadership': ['leadership', 'command', 'supervision', 'management', 'authority'],
            'training': ['training', 'education', 'instruction', 'learning', 'development']
        }
        
        # Common military terms for complexity assessment
        self.military_terms = set([
            'tactical', 'strategic', 'operational', 'deployment', 'mission',
            'command', 'control', 'communications', 'intelligence', 'surveillance',
            'reconnaissance', 'logistics', 'maintenance', 'security', 'protocol',
            'procedure', 'standard', 'operating', 'procedure', 'sop'
        ])
    
    def assess_flashcard_quality(self, flashcard: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive quality assessment of a flashcard"""
        try:
            term = flashcard.get('term', '').strip()
            definition = flashcard.get('definition', '').strip()
            
            if not term or not definition:
                return self._create_quality_result(0.0, "Missing term or definition")
            
            # Calculate individual metrics
            length_score = self._assess_definition_length(definition)
            complexity_score = self._assess_term_complexity(term)
            clarity_score = self._assess_definition_clarity(definition)
            relevance_score = self._validate_military_context(flashcard)
            testability_score = self._assess_testability(flashcard)
            completeness_score = self._assess_completeness(flashcard)
            
            # Calculate weighted overall score
            weights = self.config.get('quality_weights', {
                'length': 0.15,
                'complexity': 0.10,
                'clarity': 0.25,
                'relevance': 0.20,
                'testability': 0.15,
                'completeness': 0.15
            })
            
            overall_score = (
                length_score * weights['length'] +
                complexity_score * weights['complexity'] +
                clarity_score * weights['clarity'] +
                relevance_score * weights['relevance'] +
                testability_score * weights['testability'] +
                completeness_score * weights['completeness']
            )
            
            # Create quality metrics object
            metrics = QualityMetrics(
                definition_length_score=length_score,
                term_complexity_score=complexity_score,
                definition_clarity_score=clarity_score,
                military_relevance_score=relevance_score,
                testability_score=testability_score,
                completeness_score=completeness_score,
                overall_score=overall_score
            )
            
            return {
                'overall_score': round(overall_score, 3),
                'metrics': {
                    'definition_length_score': round(length_score, 3),
                    'term_complexity_score': round(complexity_score, 3),
                    'definition_clarity_score': round(clarity_score, 3),
                    'military_relevance_score': round(relevance_score, 3),
                    'testability_score': round(testability_score, 3),
                    'completeness_score': round(completeness_score, 3)
                },
                'difficulty_level': self.calculate_difficulty_level(flashcard),
                'recommendations': self._generate_recommendations(metrics),
                'status': 'valid' if overall_score >= 0.6 else 'needs_improvement'
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing flashcard quality: {e}")
            return self._create_quality_result(0.0, f"Assessment error: {str(e)}")
    
    def _assess_definition_length(self, definition: str) -> float:
        """Assess definition length quality"""
        length = len(definition)
        
        # Optimal length range: 50-200 characters
        min_length = self.config.get('quality_thresholds', {}).get('min_definition_length', 50)
        max_length = self.config.get('quality_thresholds', {}).get('max_definition_length', 600)
        optimal_min = 50
        optimal_max = 200
        
        if length < min_length:
            return 0.0
        elif length < optimal_min:
            return 0.3 + (length / optimal_min) * 0.4
        elif length <= optimal_max:
            return 1.0
        elif length <= max_length:
            return 1.0 - ((length - optimal_max) / (max_length - optimal_max)) * 0.3
        else:
            return 0.0
    
    def _assess_term_complexity(self, term: str) -> float:
        """Assess term complexity level"""
        words = term.lower().split()
        
        if not words:
            return 0.0
        
        # Count military-specific terms
        military_word_count = sum(1 for word in words if word in self.military_terms)
        
        # Assess word length and complexity
        avg_word_length = sum(len(word) for word in words) / len(words)
        long_words = sum(1 for word in words if len(word) > 8)
        
        # Calculate complexity score
        complexity_score = 0.0
        
        # Military term presence (positive)
        if military_word_count > 0:
            complexity_score += min(0.4, military_word_count * 0.2)
        
        # Word length (moderate complexity is good)
        if 4 <= avg_word_length <= 8:
            complexity_score += 0.3
        elif avg_word_length > 8:
            complexity_score += 0.2
        
        # Long word presence (moderate is good)
        if 1 <= long_words <= 2:
            complexity_score += 0.3
        elif long_words > 2:
            complexity_score += 0.1
        
        return min(1.0, complexity_score)
    
    def _assess_definition_clarity(self, definition: str) -> float:
        """Assess definition clarity and readability"""
        if not definition:
            return 0.0
        
        # Sentence structure analysis
        sentences = re.split(r'[.!?]+', definition)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        clarity_score = 0.0
        
        # Sentence length (optimal: 10-25 words)
        for sentence in sentences:
            word_count = len(sentence.split())
            if 10 <= word_count <= 25:
                clarity_score += 0.4
            elif 5 <= word_count <= 30:
                clarity_score += 0.2
        
        # Normalize by number of sentences
        clarity_score = min(1.0, clarity_score / len(sentences))
        
        # Check for clear structure indicators
        structure_indicators = ['is', 'are', 'refers to', 'means', 'defined as', 'consists of']
        if any(indicator in definition.lower() for indicator in structure_indicators):
            clarity_score += 0.2
        
        # Check for examples or specific details
        if re.search(r'for example|such as|including|specifically', definition.lower()):
            clarity_score += 0.1
        
        return min(1.0, clarity_score)
    
    def _validate_military_context(self, flashcard: Dict[str, Any]) -> float:
        """Validate military context relevance"""
        term = flashcard.get('term', '').lower()
        definition = flashcard.get('definition', '').lower()
        
        text = f"{term} {definition}"
        
        # Count military keyword matches
        total_matches = 0
        total_keywords = 0
        
        for category, keywords in self.military_keywords.items():
            for keyword in keywords:
                total_keywords += 1
                if keyword in text:
                    total_matches += 1
        
        # Calculate relevance score
        if total_keywords == 0:
            return 0.5  # Neutral if no keywords defined
        
        relevance_score = total_matches / total_keywords
        
        # Boost score for high relevance
        if relevance_score > 0.1:
            relevance_score = min(1.0, relevance_score * 1.5)
        
        return relevance_score
    
    def _assess_testability(self, flashcard: Dict[str, Any]) -> float:
        """Assess how well the flashcard can be tested"""
        term = flashcard.get('term', '').strip()
        definition = flashcard.get('definition', '').strip()
        
        if not term or not definition:
            return 0.0
        
        testability_score = 0.0
        
        # Term should be specific and testable
        if len(term.split()) <= 3:  # Not too long
            testability_score += 0.3
        
        # Definition should be clear and complete
        if len(definition) >= 20:  # Sufficient detail
            testability_score += 0.3
        
        # Check for clear definition structure
        if re.search(r'\b(is|are|refers to|means|defined as)\b', definition.lower()):
            testability_score += 0.2
        
        # Check for specific details that can be tested
        if re.search(r'\d+|[A-Z]{2,}', definition):  # Numbers or acronyms
            testability_score += 0.1
        
        # Check for examples or applications
        if re.search(r'for example|such as|including|used in', definition.lower()):
            testability_score += 0.1
        
        return min(1.0, testability_score)
    
    def _assess_completeness(self, flashcard: Dict[str, Any]) -> float:
        """Assess information completeness"""
        term = flashcard.get('term', '').strip()
        definition = flashcard.get('definition', '').strip()
        
        completeness_score = 0.0
        
        # Basic completeness
        if term and definition:
            completeness_score += 0.4
        
        # Definition length indicates detail level
        if len(definition) >= 50:
            completeness_score += 0.3
        elif len(definition) >= 30:
            completeness_score += 0.2
        
        # Check for additional context
        if flashcard.get('source_slide'):
            completeness_score += 0.1
        
        if flashcard.get('confidence'):
            completeness_score += 0.1
        
        # Check for specific details
        if re.search(r'\d+', definition):  # Contains numbers
            completeness_score += 0.1
        
        return min(1.0, completeness_score)
    
    def calculate_difficulty_level(self, flashcard: Dict[str, Any]) -> str:
        """Calculate difficulty level of the flashcard"""
        term = flashcard.get('term', '').lower()
        definition = flashcard.get('definition', '').lower()
        
        # Count military terms
        military_term_count = sum(1 for word in term.split() if word in self.military_terms)
        
        # Assess definition complexity
        avg_word_length = sum(len(word) for word in definition.split()) / max(1, len(definition.split()))
        long_words = sum(1 for word in definition.split() if len(word) > 8)
        
        # Calculate difficulty score
        difficulty_score = 0
        
        # Military terms (higher difficulty)
        difficulty_score += military_term_count * 2
        
        # Word length (higher difficulty)
        if avg_word_length > 7:
            difficulty_score += 2
        elif avg_word_length > 6:
            difficulty_score += 1
        
        # Long words (higher difficulty)
        difficulty_score += min(3, long_words)
        
        # Determine level
        if difficulty_score <= 2:
            return 'basic'
        elif difficulty_score <= 5:
            return 'intermediate'
        else:
            return 'advanced'
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Generate improvement recommendations based on metrics"""
        recommendations = []
        
        if metrics.definition_length_score < 0.7:
            if metrics.definition_length_score < 0.3:
                recommendations.append("Definition is too short - add more detail")
            else:
                recommendations.append("Consider expanding the definition for better clarity")
        
        if metrics.definition_clarity_score < 0.7:
            recommendations.append("Improve sentence structure and clarity")
        
        if metrics.military_relevance_score < 0.5:
            recommendations.append("Add more military context or specific terminology")
        
        if metrics.testability_score < 0.6:
            recommendations.append("Make the definition more specific and testable")
        
        if metrics.completeness_score < 0.7:
            recommendations.append("Add more complete information or examples")
        
        if not recommendations:
            recommendations.append("Flashcard quality is good")
        
        return recommendations
    
    def _create_quality_result(self, score: float, message: str) -> Dict[str, Any]:
        """Create a quality result for invalid flashcards"""
        return {
            'overall_score': score,
            'metrics': {
                'definition_length_score': 0.0,
                'term_complexity_score': 0.0,
                'definition_clarity_score': 0.0,
                'military_relevance_score': 0.0,
                'testability_score': 0.0,
                'completeness_score': 0.0
            },
            'difficulty_level': 'unknown',
            'recommendations': [message],
            'status': 'invalid'
        }


def main():
    """Test the quality assessor with sample flashcards."""
    assessor = FlashcardQualityAssessor()
    
    # Sample flashcards for testing
    sample_flashcards = [
        {
            "term": "Command and Control",
            "definition": "The exercise of authority and direction by a properly designated commander over assigned and attached forces in the accomplishment of the mission."
        },
        {
            "term": "C2",
            "definition": "Short for Command and Control."
        },
        {
            "term": "The",
            "definition": "A word."
        }
    ]
    
    print("Flashcard Quality Assessment Results:")
    print("=" * 50)
    
    for i, flashcard in enumerate(sample_flashcards, 1):
        print(f"\nFlashcard {i}:")
        print(f"Term: {flashcard['term']}")
        print(f"Definition: {flashcard['definition']}")
        
        assessment = assessor.assess_flashcard_quality(flashcard)
        print(f"Overall Score: {assessment['overall_score']}")
        print(f"Quality Level: {assessment['status']}")
        print(f"Difficulty: {assessment['difficulty_level']}")
        print(f"Issues: {', '.join(assessment['recommendations'])}")
        print(f"Recommendations: {', '.join(assessment['recommendations'])}")
    
    # Batch assessment
    print("\n" + "=" * 50)
    print("Batch Assessment Summary:")
    # The batch_assess_quality method was removed from the new_code, so this section is commented out or removed.
    # For now, we'll just print a placeholder message.
    print("Batch assessment functionality is not directly available in this version.")
    print("Please call assess_flashcard_quality for individual flashcards.")


if __name__ == "__main__":
    main()
