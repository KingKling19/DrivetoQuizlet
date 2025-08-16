#!/usr/bin/env python3
"""
Cross-Lesson Analyzer

Analyzes content similarities between lessons to enable cross-lesson context enhancement.
Detects related concepts, cross-references, and generates lesson relationship graphs.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import pickle
import numpy as np
from datetime import datetime

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    print("ERROR: scikit-learn is required. pip install scikit-learn", file=sys.stderr)
    raise

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai>=1.0.0 is required. pip install openai", file=sys.stderr)
    raise


class CrossLessonAnalyzer:
    """Analyzes relationships between lessons for context enhancement."""
    
    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize OpenAI client
        self.openai_client = OpenAI()
        
        # Data storage
        self.content_index = {}
        self.semantic_embeddings = {}
        self.lesson_relationships = {}
        self.similarity_matrix = {}
        self.cross_references = {}
        
        # Configuration
        self.relationships_file = self.config_dir / "lesson_relationships_analysis.json"
        self.similarity_file = self.config_dir / "lesson_similarity_matrix.json"
        self.cross_refs_file = self.config_dir / "cross_references.json"
        
        # Load existing data
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load existing analysis data."""
        # Load content index
        index_file = self.config_dir / "lesson_content_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    self.content_index = json.load(f)
                print(f"✓ Loaded content index with {len(self.content_index)} lessons")
            except Exception as e:
                print(f"⚠️  Could not load content index: {e}")
        
        # Load semantic embeddings
        embeddings_file = self.config_dir / "semantic_embeddings.pkl"
        if embeddings_file.exists():
            try:
                with open(embeddings_file, 'rb') as f:
                    self.semantic_embeddings = pickle.load(f)
                print(f"✓ Loaded semantic embeddings for {len(self.semantic_embeddings)} lessons")
            except Exception as e:
                print(f"⚠️  Could not load embeddings: {e}")
        
        # Load existing relationships
        if self.relationships_file.exists():
            try:
                with open(self.relationships_file, 'r', encoding='utf-8') as f:
                    self.lesson_relationships = json.load(f)
                print(f"✓ Loaded existing relationships")
            except Exception as e:
                print(f"⚠️  Could not load relationships: {e}")
        
        # Load similarity matrix
        if self.similarity_file.exists():
            try:
                with open(self.similarity_file, 'r', encoding='utf-8') as f:
                    self.similarity_matrix = json.load(f)
                print(f"✓ Loaded similarity matrix")
            except Exception as e:
                print(f"⚠️  Could not load similarity matrix: {e}")
        
        # Load cross-references
        if self.cross_refs_file.exists():
            try:
                with open(self.cross_refs_file, 'r', encoding='utf-8') as f:
                    self.cross_references = json.load(f)
                print(f"✓ Loaded cross-references")
            except Exception as e:
                print(f"⚠️  Could not load cross-references: {e}")
    
    def analyze_lesson_similarities(self):
        """Analyze similarities between all lessons."""
        if not self.content_index:
            print("⚠️  No content index available for analysis")
            return
        
        print("🔍 Analyzing lesson similarities...")
        
        lesson_ids = list(self.content_index.keys())
        similarity_matrix = {}
        
        for i, lesson_id_1 in enumerate(lesson_ids):
            similarity_matrix[lesson_id_1] = {}
            
            for j, lesson_id_2 in enumerate(lesson_ids):
                if i == j:
                    similarity_matrix[lesson_id_1][lesson_id_2] = 1.0
                else:
                    similarity = self._calculate_lesson_similarity(lesson_id_1, lesson_id_2)
                    similarity_matrix[lesson_id_1][lesson_id_2] = similarity
        
        self.similarity_matrix = similarity_matrix
        print(f"✓ Calculated similarity matrix for {len(lesson_ids)} lessons")
    
    def _calculate_lesson_similarity(self, lesson_id_1: str, lesson_id_2: str) -> float:
        """Calculate similarity between two lessons using multiple methods."""
        if lesson_id_1 not in self.content_index or lesson_id_2 not in self.content_index:
            return 0.0
        
        # Method 1: TF-IDF cosine similarity
        tfidf_similarity = self._calculate_tfidf_similarity(lesson_id_1, lesson_id_2)
        
        # Method 2: Semantic embedding similarity
        semantic_similarity = self._calculate_semantic_similarity(lesson_id_1, lesson_id_2)
        
        # Method 3: Concept overlap similarity
        concept_similarity = self._calculate_concept_overlap(lesson_id_1, lesson_id_2)
        
        # Weighted combination
        weighted_similarity = (0.4 * tfidf_similarity + 
                              0.4 * semantic_similarity + 
                              0.2 * concept_similarity)
        
        return weighted_similarity
    
    def _calculate_tfidf_similarity(self, lesson_id_1: str, lesson_id_2: str) -> float:
        """Calculate TF-IDF cosine similarity between lessons."""
        try:
            lesson1_text = self._extract_lesson_text(lesson_id_1)
            lesson2_text = self._extract_lesson_text(lesson_id_2)
            
            if not lesson1_text or not lesson2_text:
                return 0.0
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([lesson1_text, lesson2_text])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"⚠️  Error calculating TF-IDF similarity: {e}")
            return 0.0
    
    def _calculate_semantic_similarity(self, lesson_id_1: str, lesson_id_2: str) -> float:
        """Calculate semantic similarity using embeddings."""
        try:
            if lesson_id_1 not in self.semantic_embeddings or lesson_id_2 not in self.semantic_embeddings:
                return 0.0
            
            embedding1 = np.array(self.semantic_embeddings[lesson_id_1])
            embedding2 = np.array(self.semantic_embeddings[lesson_id_2])
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            return float(similarity)
        except Exception as e:
            print(f"⚠️  Error calculating semantic similarity: {e}")
            return 0.0
    
    def _calculate_concept_overlap(self, lesson_id_1: str, lesson_id_2: str) -> float:
        """Calculate similarity based on concept overlap."""
        try:
            lesson1_concepts = set(self.content_index[lesson_id_1].get("key_concepts", []))
            lesson2_concepts = set(self.content_index[lesson_id_2].get("key_concepts", []))
            
            if not lesson1_concepts or not lesson2_concepts:
                return 0.0
            
            intersection = len(lesson1_concepts.intersection(lesson2_concepts))
            union = len(lesson1_concepts.union(lesson2_concepts))
            
            if union == 0:
                return 0.0
            
            return intersection / union
        except Exception as e:
            print(f"⚠️  Error calculating concept overlap: {e}")
            return 0.0
    
    def _extract_lesson_text(self, lesson_id: str) -> str:
        """Extract all text content from a lesson."""
        if lesson_id not in self.content_index:
            return ""
        
        lesson_data = self.content_index[lesson_id]
        if not isinstance(lesson_data, dict):
            print(f"⚠️  Unexpected data type for lesson {lesson_id}: {type(lesson_data)}")
            return ""
        
        text_parts = []
        
        # Add lesson name
        text_parts.append(lesson_data.get("lesson_name", ""))
        
        # Add key concepts
        key_concepts = lesson_data.get("key_concepts", [])
        if isinstance(key_concepts, list):
            text_parts.extend(key_concepts)
        
        # Add content from presentations
        content_sources = lesson_data.get("content_sources", {})
        if isinstance(content_sources, dict):
            presentations = content_sources.get("presentations", {})
            if isinstance(presentations, dict):
                for pptx_name, pptx_data in presentations.items():
                    if isinstance(pptx_data, dict):
                        slides = pptx_data.get("slides", [])
                        if isinstance(slides, list):
                            for slide in slides:
                                if isinstance(slide, dict):
                                    text_parts.append(slide.get("title", ""))
                                    text_parts.append(slide.get("body", ""))
                                    text_parts.append(slide.get("notes", ""))
        
        return " ".join(text_parts)
    
    def _extract_related_concepts(self, lesson_id_1: str, lesson_id_2: str) -> List[str]:
        """Extract concepts that are related between two lessons."""
        try:
            lesson1_data = self.content_index.get(lesson_id_1, {})
            lesson2_data = self.content_index.get(lesson_id_2, {})
            
            if not isinstance(lesson1_data, dict) or not isinstance(lesson2_data, dict):
                return []
            
            lesson1_concepts = set(lesson1_data.get("key_concepts", []) if isinstance(lesson1_data.get("key_concepts"), list) else [])
            lesson2_concepts = set(lesson2_data.get("key_concepts", []) if isinstance(lesson2_data.get("key_concepts"), list) else [])
            
            # Find overlapping concepts
            overlapping = lesson1_concepts.intersection(lesson2_concepts)
            
            # Also find semantically similar concepts (simplified)
            similar_concepts = []
            for concept1 in lesson1_concepts:
                for concept2 in lesson2_concepts:
                    if concept1.lower() in concept2.lower() or concept2.lower() in concept1.lower():
                        if concept1 not in overlapping and concept2 not in overlapping:
                            similar_concepts.extend([concept1, concept2])
            
            # Combine and deduplicate
            all_related = list(overlapping) + list(set(similar_concepts))
            return all_related[:10]  # Limit to top 10
        except Exception as e:
            print(f"⚠️  Error extracting related concepts: {e}")
            return []
    
    def generate_lesson_relationships(self):
        """Generate comprehensive lesson relationships."""
        if not self.similarity_matrix:
            print("⚠️  No similarity matrix available. Run analyze_lesson_similarities() first.")
            return
        
        print("🔍 Generating lesson relationships...")
        
        relationships = {}
        
        for lesson_id in self.similarity_matrix.keys():
            relationships[lesson_id] = {
                "prerequisites": [],
                "related_lessons": [],
                "complementary_lessons": [],
                "relationship_scores": {}
            }
            
            # Get similarities with other lessons
            similarities = self.similarity_matrix[lesson_id]
            related_lessons = []
            
            for other_id, similarity in similarities.items():
                if other_id != lesson_id and similarity > 0.1:  # Threshold for meaningful relationship
                    related_lessons.append({
                        "lesson_id": other_id,
                        "similarity_score": similarity,
                        "relationship_type": self._determine_relationship_type(lesson_id, other_id, similarity),
                        "related_concepts": self._extract_related_concepts(lesson_id, other_id)
                    })
            
            # Sort by similarity score
            related_lessons.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # Store detailed related lessons information
            relationships[lesson_id]["related_lessons"] = related_lessons
            
            # Also maintain the old format for compatibility
            for rel in related_lessons:
                rel_type = rel["relationship_type"]
                if rel_type == "prerequisite":
                    relationships[lesson_id]["prerequisites"].append(rel["lesson_id"])
                elif rel_type == "complementary":
                    relationships[lesson_id]["complementary_lessons"].append(rel["lesson_id"])
                
                relationships[lesson_id]["relationship_scores"][rel["lesson_id"]] = rel["similarity_score"]
        
        self.lesson_relationships = relationships
    
    def _determine_relationship_type(self, lesson1_id: str, lesson2_id: str, similarity: float) -> str:
        """Determine the type of relationship between two lessons."""
        # This is a simplified classification
        # In a more sophisticated system, this could use AI analysis
        
        if similarity > 0.7:
            return "complementary"
        elif similarity > 0.5:
            return "related"
        elif similarity > 0.3:
            return "prerequisite"
        else:
            return "weakly_related"
    
    def detect_cross_references(self):
        """Detect cross-references between lessons."""
        if not self.content_index:
            print("⚠️  No content index available for cross-reference detection")
            return
        
        print("🔍 Detecting cross-references...")
        
        cross_refs = defaultdict(list)
        
        for lesson_id_1 in self.content_index.keys():
            lesson1_data = self.content_index[lesson_id_1]
            lesson1_text = self._extract_lesson_text(lesson_id_1)
            
            for lesson_id_2 in self.content_index.keys():
                if lesson_id_1 == lesson_id_2:
                    continue
                
                lesson2_data = self.content_index[lesson_id_2]
                lesson2_name = lesson2_data.get("lesson_name", lesson_id_2)
                
                # Simple keyword-based cross-reference detection
                if lesson2_name.lower() in lesson1_text.lower():
                    cross_refs[lesson_id_1].append({
                        "target_lesson": lesson_id_2,
                        "reference_type": "lesson_mention",
                        "context": f"Mentions {lesson2_name}",
                        "detailed_references": [{
                            "context_lesson1": f"References {lesson2_name}",
                            "context_lesson2": f"Referenced by {lesson1_data.get('lesson_name', lesson_id_1)}"
                        }]
                    })
        
        self.cross_references = dict(cross_refs)
        print(f"✓ Detected cross-references for {len(cross_refs)} lessons")
    
    def get_context_recommendations(self, lesson_id: str, max_context_lessons: int = 3) -> List[Dict[str, Any]]:
        """Get context enhancement recommendations for a lesson."""
        if lesson_id not in self.lesson_relationships:
            return []
        
        recommendations = []
        relationships = self.lesson_relationships[lesson_id]
        
        # Get related lessons sorted by similarity
        all_related = []
        for other_id, score in relationships["relationship_scores"].items():
            all_related.append({
                "lesson_id": other_id,
                "similarity_score": score,
                "relationship_type": self._get_relationship_type(other_id, relationships)
            })
        
        # Sort by similarity score
        all_related.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Take top recommendations
        for rel in all_related[:max_context_lessons]:
            recommendations.append({
                "lesson_id": rel["lesson_id"],
                "context_weight": rel["similarity_score"],
                "relationship_type": rel["relationship_type"],
                "recommended_context": self._get_context_snippets(lesson_id, rel["lesson_id"])
            })
        
        return recommendations
    
    def _get_relationship_type(self, other_id: str, relationships: Dict) -> str:
        """Get relationship type for a specific lesson."""
        if other_id in relationships["prerequisites"]:
            return "prerequisite"
        elif other_id in relationships["complementary_lessons"]:
            return "complementary"
        elif other_id in relationships["related_lessons"]:
            return "related"
        else:
            return "weakly_related"
    
    def _get_context_snippets(self, source_lesson: str, target_lesson: str) -> List[str]:
        """Get relevant context snippets from target lesson."""
        if source_lesson not in self.cross_references:
            return []
        
        snippets = []
        for ref in self.cross_references[source_lesson]:
            if ref["target_lesson"] == target_lesson:
                for detailed_ref in ref.get("detailed_references", []):
                    if detailed_ref.get("context_lesson2"):
                        snippets.append(detailed_ref["context_lesson2"])
        
        return snippets[:3]  # Limit to 3 snippets
    
    def save_analysis_results(self):
        """Save all analysis results to files."""
        # Save lesson relationships
        with open(self.relationships_file, 'w', encoding='utf-8') as f:
            json.dump(self.lesson_relationships, f, indent=2, ensure_ascii=False)
        
        # Save similarity matrix
        with open(self.similarity_file, 'w', encoding='utf-8') as f:
            json.dump(self.similarity_matrix, f, indent=2, ensure_ascii=False)
        
        # Save cross-references
        with open(self.cross_refs_file, 'w', encoding='utf-8') as f:
            json.dump(self.cross_references, f, indent=2, ensure_ascii=False)
        
        print("✓ Saved all analysis results")
    
    def run_full_analysis(self):
        """Run complete cross-lesson analysis."""
        print("🚀 Starting full cross-lesson analysis...")
        
        # Step 1: Analyze similarities
        self.analyze_lesson_similarities()
        
        # Step 2: Generate relationships
        self.generate_lesson_relationships()
        
        # Step 3: Detect cross-references
        self.detect_cross_references()
        
        # Step 4: Save results
        self.save_analysis_results()
        
        print("✅ Cross-lesson analysis completed!")

    def get_lesson_relationships(self, lesson_id: str) -> Dict[str, Any]:
        """Get relationships for a specific lesson."""
        return self.lesson_relationships.get(lesson_id, {})
    
    def get_related_lessons(self, lesson_id: str, max_lessons: int = 3) -> List[str]:
        """Get related lessons for a given lesson."""
        try:
            if lesson_id not in self.similarity_matrix:
                return []
            
            # Get similarity scores for this lesson
            similarities = self.similarity_matrix.get(lesson_id, {})
            
            # Sort by similarity score (descending)
            sorted_lessons = sorted(
                similarities.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Return top related lessons
            related_lessons = [lesson for lesson, score in sorted_lessons[:max_lessons] if lesson != lesson_id]
            
            return related_lessons
            
        except Exception as e:
            print(f"Error getting related lessons for {lesson_id}: {e}")
            return []
    
    def get_lesson_context(self, lesson_ids: List[str]) -> Dict[str, Any]:
        """Get context data for a list of lessons."""
        try:
            context_data = {}
            
            for lesson_id in lesson_ids:
                if lesson_id in self.content_index:
                    lesson_data = self.content_index[lesson_id]
                    
                    # Extract key information
                    context_data[lesson_id] = {
                        'title': lesson_data.get('title', lesson_id),
                        'keywords': lesson_data.get('keywords', []),
                        'topics': lesson_data.get('topics', []),
                        'summary': lesson_data.get('summary', ''),
                        'content_snippets': lesson_data.get('content_snippets', [])[:5]  # Top 5 snippets
                    }
            
            return context_data
            
        except Exception as e:
            print(f"Error getting lesson context: {e}")
            return {}


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Cross-Lesson Analyzer")
    parser.add_argument("--config-dir", default="config", help="Configuration directory")
    parser.add_argument("--full-analysis", action="store_true", help="Run full analysis")
    parser.add_argument("--analyze-similarities", action="store_true", help="Analyze lesson similarities")
    parser.add_argument("--generate-relationships", action="store_true", help="Generate lesson relationships")
    parser.add_argument("--detect-cross-refs", action="store_true", help="Detect cross-references")
    parser.add_argument("--save-results", action="store_true", help="Save analysis results")
    
    args = parser.parse_args()
    
    analyzer = CrossLessonAnalyzer(Path(args.config_dir))
    
    if args.full_analysis:
        analyzer.run_full_analysis()
    else:
        if args.analyze_similarities:
            analyzer.analyze_lesson_similarities()
        
        if args.generate_relationships:
            analyzer.generate_lesson_relationships()
        
        if args.detect_cross_refs:
            analyzer.detect_cross_references()
        
        if args.save_results:
            analyzer.save_analysis_results()


if __name__ == "__main__":
    main()
