#!/usr/bin/env python3
"""
Context Optimizer

Implements advanced context weighting and optimization algorithms for the cross-lesson context system.
Provides intelligent context selection, adaptive context window sizing, and context quality assessment.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pickle
import numpy as np

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai>=1.0.0 is required. pip install openai", file=sys.stderr)
    raise

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    print("ERROR: scikit-learn is required. pip install scikit-learn", file=sys.stderr)
    raise


class ContextOptimizer:
    """Optimizes context selection for maximum relevance and efficiency."""
    
    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize OpenAI client
        self.openai_client = OpenAI()
        
        # Load cross-lesson data
        self.cross_lesson_data = self.load_cross_lesson_data()
        
        # Optimization configuration
        self.optimization_config = {
            "max_context_length": 2000,
            "min_similarity_threshold": 0.3,
            "max_related_lessons": 3,
            "context_weight_factors": {
                "semantic_similarity": 0.4,
                "concept_overlap": 0.3,
                "lesson_relationship": 0.2,
                "content_freshness": 0.1
            },
            "adaptive_window_sizing": True,
            "quality_assessment_enabled": True
        }
    
    def load_cross_lesson_data(self) -> Dict[str, Any]:
        """Load cross-lesson analysis data."""
        data = {
            "content_index": {},
            "semantic_embeddings": {},
            "lesson_relationships": {},
            "cross_references": {}
        }
        
        try:
            # Load content index
            index_file = self.config_dir / "lesson_content_index.json"
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    data["content_index"] = json.load(f)
            
            # Load semantic embeddings
            embeddings_file = self.config_dir / "semantic_embeddings.pkl"
            if embeddings_file.exists():
                with open(embeddings_file, 'rb') as f:
                    data["semantic_embeddings"] = pickle.load(f)
            
            # Load lesson relationships
            relationships_file = self.config_dir / "lesson_relationships_analysis.json"
            if relationships_file.exists():
                with open(relationships_file, 'r', encoding='utf-8') as f:
                    data["lesson_relationships"] = json.load(f)
            
            # Load cross-references
            cross_refs_file = self.config_dir / "cross_references.json"
            if cross_refs_file.exists():
                with open(cross_refs_file, 'r', encoding='utf-8') as f:
                    data["cross_references"] = json.load(f)
            
            print(f"✓ Loaded cross-lesson data: {len(data['content_index'])} lessons indexed")
            return data
        except Exception as e:
            print(f"⚠️  Could not load cross-lesson data: {e}")
            return data
    
    def calculate_context_weight(self, source_lesson: str, target_lesson: str, content_type: str = "general") -> float:
        """Calculate weighted context relevance score."""
        try:
            # Get lesson data
            source_data = self.cross_lesson_data["content_index"].get(source_lesson, {})
            target_data = self.cross_lesson_data["content_index"].get(target_lesson, {})
            
            if not source_data or not target_data:
                return 0.0
            
            # Factor 1: Semantic similarity
            semantic_score = self._calculate_semantic_similarity(source_lesson, target_lesson)
            
            # Factor 2: Concept overlap
            concept_score = self._calculate_concept_overlap(source_data, target_data)
            
            # Factor 3: Lesson relationship strength
            relationship_score = self._calculate_relationship_strength(source_lesson, target_lesson)
            
            # Factor 4: Content freshness (based on processing timestamp)
            freshness_score = self._calculate_content_freshness(target_data)
            
            # Weighted combination
            weights = self.optimization_config["context_weight_factors"]
            weighted_score = (
                weights["semantic_similarity"] * semantic_score +
                weights["concept_overlap"] * concept_score +
                weights["lesson_relationship"] * relationship_score +
                weights["content_freshness"] * freshness_score
            )
            
            return weighted_score
            
        except Exception as e:
            print(f"⚠️  Error calculating context weight: {e}")
            return 0.0
    
    def _calculate_semantic_similarity(self, lesson1: str, lesson2: str) -> float:
        """Calculate semantic similarity using embeddings."""
        try:
            embeddings = self.cross_lesson_data["semantic_embeddings"]
            if lesson1 not in embeddings or lesson2 not in embeddings:
                return 0.0
            
            embedding1 = np.array(embeddings[lesson1])
            embedding2 = np.array(embeddings[lesson2])
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            return float(similarity)
        except Exception as e:
            print(f"⚠️  Error calculating semantic similarity: {e}")
            return 0.0
    
    def _calculate_concept_overlap(self, source_data: Dict, target_data: Dict) -> float:
        """Calculate concept overlap between lessons."""
        try:
            source_concepts = set(source_data.get("key_concepts", []))
            target_concepts = set(target_data.get("key_concepts", []))
            
            if not source_concepts or not target_concepts:
                return 0.0
            
            intersection = len(source_concepts.intersection(target_concepts))
            union = len(source_concepts.union(target_concepts))
            
            if union == 0:
                return 0.0
            
            return intersection / union
        except Exception as e:
            print(f"⚠️  Error calculating concept overlap: {e}")
            return 0.0
    
    def _calculate_relationship_strength(self, source_lesson: str, target_lesson: str) -> float:
        """Calculate relationship strength between lessons."""
        try:
            relationships = self.cross_lesson_data["lesson_relationships"]
            lesson_rels = relationships.get(source_lesson, {})
            
            # Get similarity score
            similarity = lesson_rels.get("relationship_scores", {}).get(target_lesson, 0.0)
            
            # Boost score for specific relationship types
            relationship_type = self._get_relationship_type(source_lesson, target_lesson)
            if relationship_type == "prerequisite":
                similarity *= 1.2
            elif relationship_type == "complementary":
                similarity *= 1.1
            
            return min(similarity, 1.0)
        except Exception as e:
            print(f"⚠️  Error calculating relationship strength: {e}")
            return 0.0
    
    def _get_relationship_type(self, source_lesson: str, target_lesson: str) -> str:
        """Get relationship type between lessons."""
        try:
            relationships = self.cross_lesson_data["lesson_relationships"]
            lesson_rels = relationships.get(source_lesson, {})
            
            if target_lesson in lesson_rels.get("prerequisites", []):
                return "prerequisite"
            elif target_lesson in lesson_rels.get("complementary_lessons", []):
                return "complementary"
            else:
                return "related"
        except Exception:
            return "related"
    
    def _calculate_content_freshness(self, lesson_data: Dict) -> float:
        """Calculate content freshness score based on processing timestamp."""
        try:
            metadata = lesson_data.get("content_metadata", {})
            created = metadata.get("created", "")
            
            if not created:
                return 0.5  # Default score for unknown freshness
            
            # Parse timestamp and calculate age
            from datetime import datetime
            created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
            age_days = (datetime.now() - created_dt).days
            
            # Score based on age (newer = higher score)
            if age_days <= 7:
                return 1.0
            elif age_days <= 30:
                return 0.8
            elif age_days <= 90:
                return 0.6
            else:
                return 0.4
        except Exception:
            return 0.5
    
    def optimize_context_selection(self, source_lesson: str, content_type: str = "general", max_context_length: int = None) -> List[Dict[str, Any]]:
        """Optimize context selection for maximum relevance."""
        if max_context_length is None:
            max_context_length = self.optimization_config["max_context_length"]
        
        try:
            # Get all available lessons
            available_lessons = list(self.cross_lesson_data["content_index"].keys())
            context_candidates = []
            
            # Calculate weights for all potential context sources
            for target_lesson in available_lessons:
                if target_lesson == source_lesson:
                    continue
                
                weight = self.calculate_context_weight(source_lesson, target_lesson, content_type)
                
                if weight >= self.optimization_config["min_similarity_threshold"]:
                    context_candidates.append({
                        "lesson_id": target_lesson,
                        "weight": weight,
                        "relationship_type": self._get_relationship_type(source_lesson, target_lesson),
                        "content_length": self._estimate_content_length(target_lesson)
                    })
            
            # Sort by weight and select optimal combination
            context_candidates.sort(key=lambda x: x["weight"], reverse=True)
            
            # Use adaptive window sizing if enabled
            if self.optimization_config["adaptive_window_sizing"]:
                selected_context = self._adaptive_context_selection(context_candidates, max_context_length)
            else:
                selected_context = self._fixed_context_selection(context_candidates, max_context_length)
            
            return selected_context
            
        except Exception as e:
            print(f"⚠️  Error optimizing context selection: {e}")
            return []
    
    def _adaptive_context_selection(self, candidates: List[Dict], max_length: int) -> List[Dict]:
        """Adaptive context selection based on content complexity and available space."""
        selected = []
        current_length = 0
        
        for candidate in candidates:
            estimated_length = candidate["content_length"]
            
            # Adaptive selection logic
            if current_length + estimated_length <= max_length:
                selected.append(candidate)
                current_length += estimated_length
            elif candidate["weight"] > 0.8:  # High-priority content
                # Try to fit by reducing other content
                if len(selected) > 0:
                    # Remove lowest weight content to make room
                    selected.sort(key=lambda x: x["weight"])
                    removed = selected.pop(0)
                    current_length -= removed["content_length"]
                    
                    if current_length + estimated_length <= max_length:
                        selected.append(candidate)
                        current_length += estimated_length
                        # Re-sort by weight
                        selected.sort(key=lambda x: x["weight"], reverse=True)
            
            # Stop if we have enough high-quality context
            if len(selected) >= self.optimization_config["max_related_lessons"]:
                break
        
        return selected
    
    def _fixed_context_selection(self, candidates: List[Dict], max_length: int) -> List[Dict]:
        """Fixed context selection based on weight ranking."""
        selected = []
        current_length = 0
        
        for candidate in candidates:
            if current_length + candidate["content_length"] <= max_length:
                selected.append(candidate)
                current_length += candidate["content_length"]
            
            if len(selected) >= self.optimization_config["max_related_lessons"]:
                break
        
        return selected
    
    def _estimate_content_length(self, lesson_id: str) -> int:
        """Estimate content length for a lesson."""
        try:
            lesson_data = self.cross_lesson_data["content_index"].get(lesson_id, {})
            
            # Count content from different sources
            total_length = 0
            
            content_sources = lesson_data.get("content_sources", {})
            for source_type, source_data in content_sources.items():
                if isinstance(source_data, dict):
                    text_content = source_data.get("text_content", "")
                    total_length += len(text_content)
            
            return total_length
        except Exception:
            return 1000  # Default estimate
    
    def assess_context_quality(self, selected_context: List[Dict]) -> Dict[str, Any]:
        """Assess the quality of selected context."""
        if not self.optimization_config["quality_assessment_enabled"]:
            return {"enabled": False}
        
        try:
            if not selected_context:
                return {
                    "enabled": True,
                    "quality_score": 0.0,
                    "coverage_score": 0.0,
                    "diversity_score": 0.0,
                    "recommendations": ["No context selected"]
                }
            
            # Calculate quality metrics
            weights = [ctx["weight"] for ctx in selected_context]
            avg_weight = np.mean(weights) if weights else 0.0
            
            # Coverage score (how much context we have)
            total_length = sum(ctx["content_length"] for ctx in selected_context)
            coverage_score = min(total_length / 5000, 1.0)  # Normalize to 5000 chars
            
            # Diversity score (different relationship types)
            relationship_types = set(ctx["relationship_type"] for ctx in selected_context)
            diversity_score = len(relationship_types) / 3  # Normalize to 3 types max
            
            # Overall quality score
            quality_score = (0.5 * avg_weight + 0.3 * coverage_score + 0.2 * diversity_score)
            
            # Generate recommendations
            recommendations = []
            if avg_weight < 0.5:
                recommendations.append("Consider increasing similarity threshold for better context relevance")
            if coverage_score < 0.3:
                recommendations.append("Context coverage is low - consider expanding context window")
            if diversity_score < 0.5:
                recommendations.append("Context diversity is low - consider including different lesson types")
            
            return {
                "enabled": True,
                "quality_score": quality_score,
                "coverage_score": coverage_score,
                "diversity_score": diversity_score,
                "average_weight": avg_weight,
                "total_context_length": total_length,
                "relationship_types": list(relationship_types),
                "recommendations": recommendations
            }
            
        except Exception as e:
            print(f"⚠️  Error assessing context quality: {e}")
            return {"enabled": True, "error": str(e)}
    
    def get_optimized_context(self, source_lesson: str, content_type: str = "general") -> Dict[str, Any]:
        """Get optimized context with quality assessment."""
        print(f"🔍 Optimizing context for lesson: {source_lesson}")
        
        # Optimize context selection
        selected_context = self.optimize_context_selection(source_lesson, content_type)
        
        # Assess quality
        quality_assessment = self.assess_context_quality(selected_context)
        
        # Prepare context content
        context_content = self._prepare_context_content(selected_context)
        
        # Advanced Phase 3 features
        context_analysis = self._analyze_context_patterns(selected_context)
        adaptive_recommendations = self._generate_adaptive_recommendations(source_lesson, selected_context)
        
        result = {
            "source_lesson": source_lesson,
            "content_type": content_type,
            "selected_context": selected_context,
            "context_content": context_content,
            "quality_assessment": quality_assessment,
            "context_analysis": context_analysis,
            "adaptive_recommendations": adaptive_recommendations,
            "optimization_config": self.optimization_config,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"✓ Context optimization complete:")
        print(f"  - Selected {len(selected_context)} context sources")
        print(f"  - Quality score: {quality_assessment.get('quality_score', 0):.2f}")
        print(f"  - Total context length: {len(context_content)} characters")
        print(f"  - Context patterns: {len(context_analysis.get('patterns', []))}")
        
        return result
    
    def _analyze_context_patterns(self, selected_context: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in selected context for advanced insights."""
        try:
            patterns = {
                "concept_clusters": {},
                "temporal_relationships": [],
                "content_complexity": {},
                "relationship_strength_distribution": [],
                "context_coverage_analysis": {}
            }
            
            # Analyze concept clusters
            concept_lessons = {}
            for ctx in selected_context:
                lesson_id = ctx["lesson_id"]
                lesson_data = self.cross_lesson_data["content_index"].get(lesson_id, {})
                concepts = lesson_data.get("key_concepts", [])
                
                for concept in concepts:
                    if concept not in concept_lessons:
                        concept_lessons[concept] = []
                    concept_lessons[concept].append({
                        "lesson_id": lesson_id,
                        "weight": ctx["weight"],
                        "relationship_type": ctx["relationship_type"]
                    })
            
            # Find concept clusters (concepts appearing in multiple lessons)
            for concept, lessons in concept_lessons.items():
                if len(lessons) > 1:
                    patterns["concept_clusters"][concept] = {
                        "frequency": len(lessons),
                        "lessons": lessons,
                        "avg_weight": np.mean([l["weight"] for l in lessons])
                    }
            
            # Analyze relationship strength distribution
            weights = [ctx["weight"] for ctx in selected_context]
            if weights:
                patterns["relationship_strength_distribution"] = {
                    "mean": np.mean(weights),
                    "std": np.std(weights),
                    "min": np.min(weights),
                    "max": np.max(weights),
                    "quartiles": np.percentile(weights, [25, 50, 75]).tolist()
                }
            
            # Analyze content complexity
            for ctx in selected_context:
                lesson_id = ctx["lesson_id"]
                lesson_data = self.cross_lesson_data["content_index"].get(lesson_id, {})
                
                # Estimate complexity based on content length and concept count
                concepts = lesson_data.get("key_concepts", [])
                content_length = ctx["content_length"]
                
                complexity_score = (len(concepts) * 0.3 + min(content_length / 1000, 1.0) * 0.7)
                patterns["content_complexity"][lesson_id] = {
                    "complexity_score": complexity_score,
                    "concept_count": len(concepts),
                    "content_length": content_length
                }
            
            # Context coverage analysis
            total_concepts = 0
            covered_concepts = set()
            for ctx in selected_context:
                lesson_id = ctx["lesson_id"]
                lesson_data = self.cross_lesson_data["content_index"].get(lesson_id, {})
                concepts = lesson_data.get("key_concepts", [])
                total_concepts += len(concepts)
                covered_concepts.update(concepts)
            
            patterns["context_coverage_analysis"] = {
                "total_concepts": total_concepts,
                "unique_concepts": len(covered_concepts),
                "concept_diversity": len(covered_concepts) / max(total_concepts, 1),
                "concept_overlap": total_concepts - len(covered_concepts)
            }
            
            return patterns
            
        except Exception as e:
            print(f"⚠️  Error analyzing context patterns: {e}")
            return {"error": str(e)}
    
    def _generate_adaptive_recommendations(self, source_lesson: str, selected_context: List[Dict]) -> List[str]:
        """Generate adaptive recommendations based on context analysis."""
        recommendations = []
        
        try:
            # Analyze current context
            weights = [ctx["weight"] for ctx in selected_context]
            relationship_types = [ctx["relationship_type"] for ctx in selected_context]
            
            # Weight-based recommendations
            if weights and np.mean(weights) < 0.6:
                recommendations.append("Consider lowering similarity threshold to include more relevant context")
            
            if weights and np.max(weights) < 0.8:
                recommendations.append("No high-strength relationships found - consider expanding lesson scope")
            
            # Diversity-based recommendations
            if len(set(relationship_types)) < 2:
                recommendations.append("Context lacks diversity - consider including different relationship types")
            
            # Coverage-based recommendations
            total_length = sum(ctx["content_length"] for ctx in selected_context)
            if total_length < 2000:
                recommendations.append("Context coverage is limited - consider expanding context window")
            
            # Concept-based recommendations
            concept_lessons = {}
            for ctx in selected_context:
                lesson_id = ctx["lesson_id"]
                lesson_data = self.cross_lesson_data["content_index"].get(lesson_id, {})
                concepts = lesson_data.get("key_concepts", [])
                for concept in concepts:
                    if concept not in concept_lessons:
                        concept_lessons[concept] = []
                    concept_lessons[concept].append(lesson_id)
            
            # Find frequently occurring concepts
            frequent_concepts = [concept for concept, lessons in concept_lessons.items() if len(lessons) > 1]
            if frequent_concepts:
                recommendations.append(f"Concepts {frequent_concepts[:3]} appear frequently - consider creating unified definitions")
            
            # Adaptive window sizing recommendations
            if self.optimization_config["adaptive_window_sizing"]:
                if len(selected_context) < 2:
                    recommendations.append("Adaptive window sizing suggests expanding context for better coverage")
                elif len(selected_context) > 5:
                    recommendations.append("Consider reducing context window to focus on most relevant content")
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️  Error generating adaptive recommendations: {e}")
            return ["Error generating recommendations"]
    
    def optimize_for_content_type(self, source_lesson: str, content_type: str) -> Dict[str, Any]:
        """Optimize context specifically for different content types."""
        try:
            # Adjust optimization parameters based on content type
            original_config = self.optimization_config.copy()
            
            if content_type == "definitions":
                # For definitions, prioritize high similarity and concept overlap
                self.optimization_config["context_weight_factors"]["semantic_similarity"] = 0.5
                self.optimization_config["context_weight_factors"]["concept_overlap"] = 0.4
                self.optimization_config["context_weight_factors"]["lesson_relationship"] = 0.1
                self.optimization_config["min_similarity_threshold"] = 0.4
                
            elif content_type == "procedures":
                # For procedures, prioritize prerequisite relationships
                self.optimization_config["context_weight_factors"]["lesson_relationship"] = 0.4
                self.optimization_config["context_weight_factors"]["semantic_similarity"] = 0.3
                self.optimization_config["context_weight_factors"]["concept_overlap"] = 0.3
                
            elif content_type == "concepts":
                # For concepts, balance all factors
                self.optimization_config["context_weight_factors"]["semantic_similarity"] = 0.4
                self.optimization_config["context_weight_factors"]["concept_overlap"] = 0.3
                self.optimization_config["context_weight_factors"]["lesson_relationship"] = 0.2
                self.optimization_config["context_weight_factors"]["content_freshness"] = 0.1
            
            # Get optimized context with adjusted parameters
            result = self.get_optimized_context(source_lesson, content_type)
            
            # Restore original configuration
            self.optimization_config = original_config
            
            return result
            
        except Exception as e:
            print(f"⚠️  Error optimizing for content type: {e}")
            return {"error": str(e)}
    
    def get_context_evolution_analysis(self, source_lesson: str) -> Dict[str, Any]:
        """Analyze how context has evolved over time for a lesson."""
        try:
            evolution_data = {
                "lesson_id": source_lesson,
                "context_history": [],
                "trends": {},
                "recommendations": []
            }
            
            # This would typically integrate with a database or file system
            # that tracks context usage over time. For now, we'll provide
            # a framework for future implementation.
            
            # Analyze current context patterns
            current_context = self.get_optimized_context(source_lesson)
            evolution_data["current_context"] = {
                "context_sources": len(current_context["selected_context"]),
                "quality_score": current_context["quality_assessment"].get("quality_score", 0),
                "context_length": len(current_context["context_content"])
            }
            
            # Generate evolution recommendations
            quality_score = current_context["quality_assessment"].get("quality_score", 0)
            if quality_score < 0.6:
                evolution_data["recommendations"].append("Context quality has declined - review lesson relationships")
            
            return evolution_data
            
        except Exception as e:
            print(f"⚠️  Error analyzing context evolution: {e}")
            return {"error": str(e)}
    
    def _prepare_context_content(self, selected_context: List[Dict]) -> str:
        """Prepare formatted context content from selected sources."""
        content_parts = []
        
        for ctx in selected_context:
            lesson_id = ctx["lesson_id"]
            lesson_data = self.cross_lesson_data["content_index"].get(lesson_id, {})
            
            if not lesson_data:
                continue
            
            lesson_name = lesson_data.get("lesson_name", lesson_id)
            weight = ctx["weight"]
            rel_type = ctx["relationship_type"]
            
            # Add lesson header
            content_parts.append(f"## Related Lesson: {lesson_name}")
            content_parts.append(f"**Relevance Score:** {weight:.2f} | **Type:** {rel_type}")
            
            # Add key concepts
            key_concepts = lesson_data.get("key_concepts", [])
            if key_concepts:
                content_parts.append("### Key Concepts:")
                for concept in key_concepts[:5]:  # Limit to top 5
                    content_parts.append(f"- {concept}")
            
            # Add content snippets
            content_sources = lesson_data.get("content_sources", {})
            if isinstance(content_sources, dict):
                presentations = content_sources.get("presentations", {})
                if isinstance(presentations, dict):
                    for pptx_name, pptx_data in list(presentations.items())[:1]:
                        if isinstance(pptx_data, dict):
                            slides = pptx_data.get("slides", [])
                            if isinstance(slides, list):
                                for slide in slides[:2]:  # Limit to 2 slides
                                    if isinstance(slide, dict):
                                        title = slide.get("title", "")
                                        body = slide.get("body", "")
                                        if title and body:
                                            content_parts.append(f"### {title}")
                                            content_parts.append(body[:300] + "..." if len(body) > 300 else body)
            
            content_parts.append("")  # Spacer
        
        return "\n".join(content_parts)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Context Optimizer for Cross-Lesson Analysis")
    parser.add_argument("--config-dir", default="config", help="Configuration directory")
    parser.add_argument("--source-lesson", required=True, help="Source lesson ID")
    parser.add_argument("--content-type", default="general", help="Content type for optimization")
    parser.add_argument("--max-length", type=int, default=2000, help="Maximum context length")
    parser.add_argument("--output-file", help="Output file for results")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = ContextOptimizer(Path(args.config_dir))
    
    # Get optimized context
    result = optimizer.get_optimized_context(args.source_lesson, args.content_type)
    
    # Save results if output file specified
    if args.output_file:
        output_path = Path(args.output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✓ Results saved to: {output_path}")
    
    # Print summary
    print(f"\n📊 Context Optimization Summary:")
    print(f"  Source Lesson: {result['source_lesson']}")
    print(f"  Context Sources: {len(result['selected_context'])}")
    print(f"  Quality Score: {result['quality_assessment'].get('quality_score', 0):.2f}")
    print(f"  Context Length: {len(result['context_content'])} characters")
    
    if result['quality_assessment'].get('recommendations'):
        print(f"  Recommendations:")
        for rec in result['quality_assessment']['recommendations']:
            print(f"    - {rec}")


if __name__ == "__main__":
    main()
