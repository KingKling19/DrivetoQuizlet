#!/usr/bin/env python3
"""
Lesson Content Indexer for Cross-Lesson Context System

Indexes and fingerprints lesson content for similarity analysis and 
relationship mapping across different lessons.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict
import hashlib

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("WARNING: scikit-learn and numpy required for advanced similarity analysis")
    np = None
    TfidfVectorizer = None
    cosine_similarity = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("WARNING: sentence-transformers not available. Using basic similarity analysis.")
    SentenceTransformer = None

class LessonContentIndexer:
    """Indexes lesson content for cross-lesson context analysis"""
    
    def __init__(self, lessons_dir: str = "lessons", index_dir: str = "config"):
        self.lessons_dir = Path(lessons_dir)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        # Content storage
        self.lesson_index = {}
        self.content_fingerprints = {}
        self.topic_keywords = defaultdict(set)
        self.lesson_relationships = defaultdict(list)
        
        # Analysis tools
        self.tfidf_vectorizer = None
        self.sentence_model = None
        self.content_vectors = {}
        
        # Load existing index if available
        self.load_index()
        
    def load_index(self) -> bool:
        """Load existing content index from disk"""
        index_file = self.index_dir / "lesson_content_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    data = json.load(f)
                    self.lesson_index = data.get('lesson_index', {})
                    self.content_fingerprints = data.get('content_fingerprints', {})
                    
                    # Convert lists back to sets for topic_keywords
                    topic_keywords_raw = data.get('topic_keywords', {})
                    self.topic_keywords = defaultdict(set)
                    for lesson, keywords in topic_keywords_raw.items():
                        self.topic_keywords[lesson] = set(keywords) if isinstance(keywords, list) else set()
                    
                    self.lesson_relationships = defaultdict(list, data.get('lesson_relationships', {}))
                print(f"✓ Loaded existing index with {len(self.lesson_index)} lessons")
                return True
            except Exception as e:
                print(f"WARNING: Could not load existing index: {e}")
        return False
    
    def save_index(self) -> bool:
        """Save content index to disk"""
        try:
            index_file = self.index_dir / "lesson_content_index.json"
            # Convert sets to lists for JSON serialization
            topic_keywords_serializable = {}
            for lesson, keywords in self.topic_keywords.items():
                topic_keywords_serializable[lesson] = list(keywords) if isinstance(keywords, set) else keywords
            
            data = {
                'lesson_index': self.lesson_index,
                'content_fingerprints': self.content_fingerprints,
                'topic_keywords': topic_keywords_serializable,
                'lesson_relationships': dict(self.lesson_relationships),
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            }
            with open(index_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"✓ Saved index with {len(self.lesson_index)} lessons")
            return True
        except Exception as e:
            print(f"ERROR: Could not save index: {e}")
            return False
    
    def index_all_lessons(self) -> Dict[str, Any]:
        """Index all lessons in the lessons directory"""
        if not self.lessons_dir.exists():
            print(f"ERROR: Lessons directory {self.lessons_dir} not found")
            return {}
        
        print(f"🔍 Indexing lessons in {self.lessons_dir}")
        indexed_lessons = []
        
        for lesson_dir in self.lessons_dir.iterdir():
            if lesson_dir.is_dir() and not lesson_dir.name.startswith('.'):
                try:
                    lesson_data = self.index_lesson(lesson_dir)
                    if lesson_data:
                        indexed_lessons.append(lesson_dir.name)
                        print(f"✓ Indexed lesson: {lesson_dir.name}")
                except Exception as e:
                    print(f"ERROR indexing {lesson_dir.name}: {e}")
        
        # Build cross-references after all lessons are indexed
        self._build_cross_references()
        
        # Save the updated index
        self.save_index()
        
        result = {
            'indexed_lessons': indexed_lessons,
            'total_lessons': len(indexed_lessons),
            'content_fingerprints': len(self.content_fingerprints),
            'relationships': len(self.lesson_relationships)
        }
        
        print(f"📊 Indexing complete: {result}")
        return result
    
    def index_lesson(self, lesson_path: Path) -> Optional[Dict[str, Any]]:
        """Index a single lesson directory"""
        lesson_name = lesson_path.name
        
        # Extract content from different sources
        content_data = {
            'lesson_name': lesson_name,
            'path': str(lesson_path),
            'indexed_at': datetime.now().isoformat(),
            'content_sources': {},
            'topics': set(),
            'key_terms': set(),
            'concepts': set()
        }
        
        # Process different content types
        content_data.update(self._extract_presentation_content(lesson_path))
        content_data.update(self._extract_notes_content(lesson_path))
        content_data.update(self._extract_audio_content(lesson_path))
        content_data.update(self._extract_processed_content(lesson_path))
        
        # Generate content fingerprint
        fingerprint = self._generate_content_fingerprint(content_data)
        content_data['fingerprint'] = fingerprint
        self.content_fingerprints[lesson_name] = fingerprint
        
        # Extract topics and keywords
        self._extract_topics_and_keywords(lesson_name, content_data)
        
        # Store in index
        self.lesson_index[lesson_name] = content_data
        
        return content_data
    
    def _extract_presentation_content(self, lesson_path: Path) -> Dict[str, Any]:
        """Extract content from PowerPoint presentations"""
        presentations_dir = lesson_path / "presentations"
        content = {'presentations': []}
        
        if presentations_dir.exists():
            for pptx_file in presentations_dir.glob("*.pptx"):
                try:
                    # For now, just record the file info
                    # Full extraction would require python-pptx
                    content['presentations'].append({
                        'file': pptx_file.name,
                        'size': pptx_file.stat().st_size,
                        'modified': pptx_file.stat().st_mtime
                    })
                except Exception as e:
                    print(f"WARNING: Could not process {pptx_file.name}: {e}")
        
        return content
    
    def _extract_notes_content(self, lesson_path: Path) -> Dict[str, Any]:
        """Extract content from handwritten notes"""
        notes_dir = lesson_path / "notes"
        content = {'notes': []}
        
        if notes_dir.exists():
            for note_file in notes_dir.glob("*.png"):
                try:
                    content['notes'].append({
                        'file': note_file.name,
                        'size': note_file.stat().st_size,
                        'modified': note_file.stat().st_mtime
                    })
                except Exception as e:
                    print(f"WARNING: Could not process {note_file.name}: {e}")
        
        return content
    
    def _extract_audio_content(self, lesson_path: Path) -> Dict[str, Any]:
        """Extract content from audio files"""
        audio_dir = lesson_path / "audio"
        content = {'audio': []}
        
        if audio_dir.exists():
            for audio_file in audio_dir.glob("*.m4a"):
                try:
                    content['audio'].append({
                        'file': audio_file.name,
                        'size': audio_file.stat().st_size,
                        'modified': audio_file.stat().st_mtime
                    })
                except Exception as e:
                    print(f"WARNING: Could not process {audio_file.name}: {e}")
        
        return content
    
    def _extract_processed_content(self, lesson_path: Path) -> Dict[str, Any]:
        """Extract content from processed output files"""
        processed_dir = lesson_path / "processed"
        output_dir = lesson_path / "output"
        content = {'processed_content': {}}
        
        # Check for processed flashcards
        for output_dir in [processed_dir, output_dir]:
            if output_dir and output_dir.exists():
                for json_file in output_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            content['processed_content'][json_file.name] = {
                                'type': self._determine_content_type(data),
                                'size': len(data) if isinstance(data, list) else 1,
                                'file_size': json_file.stat().st_size
                            }
                    except Exception as e:
                        print(f"WARNING: Could not process {json_file.name}: {e}")
        
        return content
    
    def _determine_content_type(self, data: Any) -> str:
        """Determine the type of processed content"""
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                if 'term' in data[0] and 'definition' in data[0]:
                    return 'flashcards'
                elif 'text' in data[0] or 'content' in data[0]:
                    return 'transcription'
        return 'unknown'
    
    def _generate_content_fingerprint(self, content_data: Dict[str, Any]) -> str:
        """Generate a unique fingerprint for lesson content"""
        # Create a stable representation of the content
        fingerprint_data = {
            'lesson_name': content_data['lesson_name'],
            'presentations': len(content_data.get('presentations', [])),
            'notes': len(content_data.get('notes', [])),
            'audio': len(content_data.get('audio', [])),
            'processed': len(content_data.get('processed_content', {}))
        }
        
        # Convert to string and hash
        data_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _extract_topics_and_keywords(self, lesson_name: str, content_data: Dict[str, Any]) -> None:
        """Extract topics and keywords from lesson content"""
        # Extract from lesson name
        name_words = re.findall(r'\b[A-Z][a-z]+\b', lesson_name.replace('_', ' '))
        self.topic_keywords[lesson_name].update(name_words)
        
        # Add military-specific keywords based on lesson name patterns
        military_keywords = self._extract_military_keywords(lesson_name)
        self.topic_keywords[lesson_name].update(military_keywords)
    
    def _extract_military_keywords(self, lesson_name: str) -> Set[str]:
        """Extract military-specific keywords from lesson names"""
        keywords = set()
        lesson_lower = lesson_name.lower()
        
        # Common military terms
        military_terms = {
            'ada': ['Air Defense Artillery', 'ADA'],
            'tlp': ['Troop Leading Procedures', 'TLP'],
            'operations': ['Operations', 'OPORD', 'Mission'],
            'degraded': ['Degraded Operations', 'CBRN', 'EW'],
            'environment': ['Operational Environment', 'Battlefield'],
            'conduct': ['Conduct', 'Execute', 'Perform'],
            'effectively': ['Effective', 'Efficient', 'Performance']
        }
        
        for key, terms in military_terms.items():
            if key in lesson_lower:
                keywords.update(terms)
        
        return keywords
    
    def _build_cross_references(self) -> None:
        """Build cross-references between lessons"""
        lesson_names = list(self.lesson_index.keys())
        
        for i, lesson1 in enumerate(lesson_names):
            for lesson2 in lesson_names[i+1:]:
                similarity = self._calculate_lesson_similarity(lesson1, lesson2)
                if similarity > 0.1:  # Threshold for related lessons
                    self.lesson_relationships[lesson1].append({
                        'related_lesson': lesson2,
                        'similarity': similarity,
                        'relationship_type': self._determine_relationship_type(lesson1, lesson2)
                    })
                    self.lesson_relationships[lesson2].append({
                        'related_lesson': lesson1,
                        'similarity': similarity,
                        'relationship_type': self._determine_relationship_type(lesson2, lesson1)
                    })
    
    def _calculate_lesson_similarity(self, lesson1: str, lesson2: str) -> float:
        """Calculate similarity between two lessons"""
        keywords1 = self.topic_keywords.get(lesson1, set())
        keywords2 = self.topic_keywords.get(lesson2, set())
        
        if not keywords1 or not keywords2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        return intersection / union if union > 0 else 0.0
    
    def _determine_relationship_type(self, lesson1: str, lesson2: str) -> str:
        """Determine the type of relationship between lessons"""
        # Simple heuristic based on lesson names
        name1_lower = lesson1.lower()
        name2_lower = lesson2.lower()
        
        if 'tlp' in name1_lower and 'tlp' in name2_lower:
            return 'procedural'
        elif 'operations' in name1_lower and 'operations' in name2_lower:
            return 'operational'
        elif any(term in name1_lower and term in name2_lower for term in ['ada', 'defense', 'artillery']):
            return 'domain-specific'
        else:
            return 'thematic'
    
    def get_related_lessons(self, lesson_name: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Get lessons related to the specified lesson"""
        relationships = self.lesson_relationships.get(lesson_name, [])
        
        # Sort by similarity score
        sorted_relationships = sorted(relationships, 
                                    key=lambda x: x['similarity'], 
                                    reverse=True)
        
        return sorted_relationships[:max_results]
    
    def get_lesson_summary(self, lesson_name: str) -> Optional[Dict[str, Any]]:
        """Get summary information for a lesson"""
        if lesson_name not in self.lesson_index:
            return None
        
        lesson_data = self.lesson_index[lesson_name]
        related_lessons = self.get_related_lessons(lesson_name, 3)
        
        return {
            'lesson_name': lesson_name,
            'fingerprint': lesson_data.get('fingerprint'),
            'content_sources': lesson_data.get('content_sources', {}),
            'topics': list(self.topic_keywords.get(lesson_name, set())),
            'related_lessons': related_lessons,
            'indexed_at': lesson_data.get('indexed_at')
        }

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Index lesson content for cross-lesson context analysis")
    parser.add_argument("--lessons-dir", default="lessons", help="Directory containing lessons")
    parser.add_argument("--index-dir", default="config", help="Directory to store index files")
    parser.add_argument("--lesson", help="Index a specific lesson")
    parser.add_argument("--summary", help="Get summary for a specific lesson")
    parser.add_argument("--related", help="Get lessons related to specified lesson")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    indexer = LessonContentIndexer(args.lessons_dir, args.index_dir)
    
    if args.lesson:
        lesson_path = Path(args.lessons_dir) / args.lesson
        if lesson_path.exists():
            result = indexer.index_lesson(lesson_path)
            indexer.save_index()
            print(f"✓ Indexed lesson: {args.lesson}")
            if args.verbose:
                print(json.dumps(result, indent=2, default=str))
        else:
            print(f"ERROR: Lesson {args.lesson} not found")
    
    elif args.summary:
        summary = indexer.get_lesson_summary(args.summary)
        if summary:
            print(json.dumps(summary, indent=2, default=str))
        else:
            print(f"ERROR: Lesson {args.summary} not found in index")
    
    elif args.related:
        related = indexer.get_related_lessons(args.related)
        if related:
            print(f"Lessons related to {args.related}:")
            for rel in related:
                print(f"  - {rel['related_lesson']} (similarity: {rel['similarity']:.3f}, type: {rel['relationship_type']})")
        else:
            print(f"No related lessons found for {args.related}")
    
    else:
        # Index all lessons
        result = indexer.index_all_lessons()
        print(f"📊 Indexing complete: {result}")

if __name__ == "__main__":
    main()