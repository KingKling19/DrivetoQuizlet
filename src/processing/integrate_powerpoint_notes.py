#!/usr/bin/env python3
"""
PowerPoint + Notes Integration for Enhanced Quizlet Flashcards

Combines PowerPoint content with handwritten notes insights to create
high-quality, comprehensive flashcards for military training.
Enhanced with cross-lesson context for better content correlation.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pickle

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env file")
except ImportError:
    print("WARNING: python-dotenv not installed", file=sys.stderr)
except Exception as e:
    print(f"WARNING: Could not load .env file: {e}", file=sys.stderr)

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package required. pip install openai", file=sys.stderr)
    sys.exit(1)

class PowerPointNotesIntegrator:
    def __init__(self, config_dir: Path = Path("config")):
        """Initialize the integrator with OpenAI client and cross-lesson context."""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
                print("✓ OpenAI client initialized successfully")
            else:
                print("ERROR: OPENAI_API_KEY not found in environment variables", file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Could not initialize OpenAI client: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Cross-lesson context configuration
        self.config_dir = Path(config_dir)
        self.cross_lesson_data = self.load_cross_lesson_data()
        
        # Context enhancement configuration
        self.context_config = {
            "max_related_lessons": 2,
            "context_weight_threshold": 0.3,
            "max_context_length": 1500,
            "include_prerequisites": True,
            "include_related_concepts": True
        }
    
    def load_cross_lesson_data(self) -> Dict[str, Any]:
        """Load cross-lesson analysis data for context enhancement."""
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
    
    def get_lesson_id_from_path(self, file_path: Path) -> str:
        """Extract lesson ID from file path."""
        # Try to find lesson directory in path
        path_parts = file_path.parts
        for i, part in enumerate(path_parts):
            if part == "lessons" and i + 1 < len(path_parts):
                return path_parts[i + 1]
        
        # Fallback: use filename without extension
        return file_path.stem
    
    def find_related_lessons(self, lesson_id: str, max_lessons: int = 2) -> List[Dict[str, Any]]:
        """Find lessons related to the current lesson for context enhancement."""
        related_lessons = []
        
        try:
            relationships = self.cross_lesson_data.get("lesson_relationships", {})
            lesson_rels = relationships.get(lesson_id, {})
            
            # Get related lessons - handle both old and new formats
            related = lesson_rels.get("related_lessons", [])
            
            # If related_lessons is a list of strings (old format), convert to new format
            if related and isinstance(related[0], str):
                # Old format - convert to new format
                related_lessons_new = []
                for rel_id in related:
                    similarity = lesson_rels.get("relationship_scores", {}).get(rel_id, 0.0)
                    related_lessons_new.append({
                        "lesson_id": rel_id,
                        "similarity_score": similarity,
                        "relationship_type": "related",
                        "related_concepts": []
                    })
                related = related_lessons_new
            
            # Sort by similarity score
            if related and isinstance(related[0], dict):
                related.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            
            # Take top related lessons
            for rel in related[:max_lessons]:
                if isinstance(rel, dict):
                    related_lesson_id = rel.get("lesson_id")
                    if related_lesson_id and related_lesson_id != lesson_id:
                        related_lessons.append(rel)
            
            return related_lessons
        except Exception as e:
            print(f"⚠️  Error finding related lessons: {e}")
            return []
    
    def extract_related_context(self, related_lessons: List[Dict[str, Any]]) -> str:
        """Extract relevant context from related lessons."""
        context_parts = []
        
        try:
            content_index = self.cross_lesson_data.get("content_index", {})
            
            for rel in related_lessons:
                if not isinstance(rel, dict):
                    continue
                    
                lesson_id = rel.get("lesson_id")
                similarity_score = rel.get("similarity_score", 0)
                related_concepts = rel.get("related_concepts", [])
                
                if lesson_id in content_index:
                    lesson_data = content_index[lesson_id]
                    if not isinstance(lesson_data, dict):
                        continue
                        
                    lesson_name = lesson_data.get("lesson_name", lesson_id)
                    
                    # Add lesson header
                    context_parts.append(f"## Related Lesson: {lesson_name} (Similarity: {similarity_score:.2f})")
                    
                    # Add key concepts
                    if related_concepts and isinstance(related_concepts, list):
                        context_parts.append("### Key Related Concepts:")
                        for concept in related_concepts[:3]:  # Limit to top 3 concepts
                            context_parts.append(f"- {concept}")
                    
                    # Add content snippets from presentations
                    content_sources = lesson_data.get("content_sources", {})
                    if isinstance(content_sources, dict):
                        presentations = content_sources.get("presentations", {})
                        if isinstance(presentations, dict):
                            for pptx_name, pptx_data in list(presentations.items())[:1]:  # Limit to 1 presentation
                                if isinstance(pptx_data, dict):
                                    slides = pptx_data.get("slides", [])
                                    if isinstance(slides, list):
                                        for slide in slides[:2]:  # Limit to 2 slides per presentation
                                            if isinstance(slide, dict):
                                                title = slide.get("title", "")
                                                body = slide.get("body", "")
                                                if title and body:
                                                    context_parts.append(f"### {title}")
                                                    context_parts.append(body[:200] + "..." if len(body) > 200 else body)
                    
                    context_parts.append("")  # Spacer
            
            context = "\n".join(context_parts)
            
            # Limit total context length
            if len(context) > self.context_config["max_context_length"]:
                context = context[:self.context_config["max_context_length"]] + "..."
            
            return context
        except Exception as e:
            print(f"⚠️  Error extracting related context: {e}")
            return ""
    
    def enhance_with_cross_lesson_context(self, lesson_id: str) -> str:
        """Enhance content with cross-lesson context."""
        try:
            related_lessons = self.find_related_lessons(lesson_id, self.context_config["max_related_lessons"])
            
            if not related_lessons:
                return ""
            
            return self.extract_related_context(related_lessons)
        except Exception as e:
            print(f"⚠️  Error enhancing with cross-lesson context: {e}")
            return ""
    
    def load_powerpoint_flashcards(self, quizlet_file: Path) -> List[Dict[str, str]]:
        """Load existing PowerPoint flashcards from Quizlet export."""
        try:
            with open(quizlet_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            flashcards = []
            # Handle both array format and object with 'cards' property
            if isinstance(data, list):
                cards_data = data
            else:
                cards_data = data.get('cards', [])
            
            for card in cards_data:
                flashcards.append({
                    'term': card.get('term', ''),
                    'definition': card.get('definition', '')
                })
            
            print(f"✓ Loaded {len(flashcards)} PowerPoint flashcards")
            return flashcards
            
        except Exception as e:
            print(f"ERROR loading PowerPoint flashcards: {e}", file=sys.stderr)
            return []
    
    def load_notes_analysis(self, notes_file: Path) -> Dict[str, Any]:
        """Load handwritten notes analysis data."""
        try:
            with open(notes_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✓ Loaded notes analysis: {data.get('total_words', 0)} words extracted")
            return data
            
        except Exception as e:
            print(f"ERROR loading notes analysis: {e}", file=sys.stderr)
            return {}
    
    def enhance_flashcards_with_notes(self, flashcards: List[Dict], notes_data: Dict, lesson_id: str = "") -> List[Dict]:
        """Enhance PowerPoint flashcards with notes insights and cross-lesson context."""
        if not notes_data:
            print("WARNING: No notes data available, returning original flashcards")
            return flashcards
        
        print("Enhancing flashcards with notes insights and cross-lesson context...")
        
        # Prepare context for enhancement
        notes_text = ""
        for filename, result in notes_data.get('results', {}).items():
            notes_text += f"\n--- {filename} ---\n"
            notes_text += result.get('enhanced_text', '')
        
        # Get cross-lesson context if available
        cross_lesson_context = ""
        if lesson_id:
            cross_lesson_context = self.enhance_with_cross_lesson_context(lesson_id)
            if cross_lesson_context:
                print(f"✓ Enhanced with cross-lesson context from {len(self.cross_lesson_data.get('content_index', {}))} lessons")
        
        # Create enhancement prompt
        enhancement_prompt = f"""
        You are a military training expert. I have PowerPoint flashcards and handwritten notes from the same lesson.
        
        LESSON TOPIC: Perform Effectively In An Operational Environment
        
        HANDWRITTEN NOTES CONTENT:
        {notes_text[:3000]}  # Limit for API
        
        NOTES STATISTICS:
        - Total images processed: {notes_data.get('total_images', 0)}
        - Average confidence: {notes_data.get('average_confidence', 0):.2f}
        - Total words extracted: {notes_data.get('total_words', 0)}
        
        EXISTING POWERPOINT FLASHCARDS:
        {json.dumps(flashcards[:10], indent=2)}  # Show first 10 for context
        
        CROSS-LESSON CONTEXT:
        {cross_lesson_context if cross_lesson_context else "No cross-lesson context available"}
        
        TASK: Enhance these flashcards by:
        1. Adding missing key concepts from the handwritten notes
        2. Improving definitions with notes context and cross-lesson insights
        3. Adding new flashcards for important notes-only content
        4. Ensuring military accuracy and test-focus
        5. Incorporating any additional insights from student notes
        6. Using cross-lesson context to avoid duplicate definitions and enhance understanding
        7. Correlating concepts across related lessons for better comprehension
        
        Return ONLY a JSON array of enhanced flashcards with 'term' and 'definition' fields.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a military training assistant specializing in creating high-quality flashcards for test preparation. Focus on accuracy, clarity, and test-relevance."},
                    {"role": "user", "content": enhancement_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            enhanced_content = response.choices[0].message.content
            
            # Try to parse the JSON response
            try:
                enhanced_flashcards = json.loads(enhanced_content)
                print(f"✓ Enhanced flashcards: {len(enhanced_flashcards)} total")
                return enhanced_flashcards
            except json.JSONDecodeError:
                print("WARNING: Could not parse enhanced flashcards, using original")
                return flashcards
                
        except Exception as e:
            print(f"ERROR enhancing flashcards: {e}", file=sys.stderr)
            return flashcards
    
    def create_comprehensive_flashcards(self, powerpoint_file: Path, notes_folder: Path, output_dir: Path) -> Dict[str, Any]:
        """Create comprehensive flashcards combining PowerPoint and notes."""
        print("=" * 60)
        print("POWERPOINT + NOTES FLASHCARD INTEGRATION")
        print("=" * 60)
        
        # Load PowerPoint flashcards
        # Look for Quizlet file in downloads folder
        quizlet_file = Path("downloads") / f"{powerpoint_file.stem}.quizlet.json"
        if not quizlet_file.exists():
            print(f"ERROR: Quizlet file not found: {quizlet_file}", file=sys.stderr)
            return {}
        
        flashcards = self.load_powerpoint_flashcards(quizlet_file)
        if not flashcards:
            print("ERROR: No flashcards loaded", file=sys.stderr)
            return {}
        
        # Load notes analysis
        notes_file = Path(f"downloads/notes/processed/{notes_folder.name}_notes_processed.json")
        if not notes_file.exists():
            print(f"ERROR: Notes analysis not found: {notes_file}", file=sys.stderr)
            return {}
        
        notes_data = self.load_notes_analysis(notes_file)
        
        # Get lesson ID for cross-lesson context
        lesson_id = self.get_lesson_id_from_path(powerpoint_file)
        print(f"📚 Processing lesson: {lesson_id}")
        
        # Enhance flashcards with notes and cross-lesson context
        enhanced_flashcards = self.enhance_flashcards_with_notes(flashcards, notes_data, lesson_id)
        
        # Create comprehensive result
        result = {
            "source_powerpoint": powerpoint_file.name,
            "source_notes": notes_folder.name,
            "lesson_id": lesson_id,
            "lesson_topic": "Perform Effectively In An Operational Environment",
            "original_flashcards": len(flashcards),
            "enhanced_flashcards": len(enhanced_flashcards),
            "notes_insights": {
                "total_images": notes_data.get('total_images', 0),
                "average_confidence": notes_data.get('average_confidence', 0),
                "total_words": notes_data.get('total_words', 0)
            },
            "cross_lesson_context": {
                "lessons_indexed": len(self.cross_lesson_data.get('content_index', {})),
                "related_lessons_found": len(self.find_related_lessons(lesson_id)),
                "context_enabled": bool(self.cross_lesson_data.get('content_index'))
            },
            "flashcards": enhanced_flashcards,
            "processing_metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_used": "gpt-4o-mini",
                "cross_lesson_enhancement": True
            }
        }
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive JSON
        json_file = output_dir / f"{powerpoint_file.stem}_enhanced_flashcards.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save Quizlet-ready TSV
        tsv_file = output_dir / f"{powerpoint_file.stem}_enhanced_flashcards.tsv"
        with open(tsv_file, 'w', encoding='utf-8') as f:
            f.write("term\tdefinition\n")
            for card in enhanced_flashcards:
                term = card.get('term', '').replace('\t', ' ').replace('\n', ' ')
                definition = card.get('definition', '').replace('\t', ' ').replace('\n', ' ')
                f.write(f"{term}\t{definition}\n")
        
        # Save summary
        summary_file = output_dir / f"{powerpoint_file.stem}_enhanced_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("ENHANCED FLASHCARDS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"PowerPoint source: {powerpoint_file.name}\n")
            f.write(f"Notes source: {notes_folder.name}\n")
            f.write(f"Lesson topic: Perform Effectively In An Operational Environment\n")
            f.write(f"Original flashcards: {len(flashcards)}\n")
            f.write(f"Enhanced flashcards: {len(enhanced_flashcards)}\n")
            f.write(f"Notes images processed: {result['notes_insights']['total_images']}\n")
            f.write(f"Notes average confidence: {result['notes_insights']['average_confidence']:.2f}\n")
            f.write(f"Notes words extracted: {result['notes_insights']['total_words']}\n\n")
            
            f.write("SAMPLE ENHANCED FLASHCARDS:\n")
            f.write("-" * 30 + "\n")
            for i, card in enumerate(enhanced_flashcards[:5]):
                f.write(f"{i+1}. {card.get('term', '')}\n")
                f.write(f"   {card.get('definition', '')}\n\n")
        
        print(f"✓ Results saved to: {output_dir}")
        print(f"  - {json_file.name}")
        print(f"  - {tsv_file.name}")
        print(f"  - {summary_file.name}")
        
        return result

def main():
    parser = argparse.ArgumentParser(description="Integrate PowerPoint and notes for enhanced flashcards")
    parser.add_argument("powerpoint_file", help="Path to PowerPoint file (.pptx)")
    parser.add_argument("notes_folder", help="Path to notes folder")
    parser.add_argument("--output-dir", default="downloads/integrated", 
                       help="Output directory for integrated files")
    
    args = parser.parse_args()
    
    powerpoint_path = Path(args.powerpoint_file)
    notes_folder = Path(args.notes_folder)
    output_dir = Path(args.output_dir)
    
    if not powerpoint_path.exists():
        print(f"ERROR: PowerPoint file not found: {powerpoint_path}", file=sys.stderr)
        sys.exit(1)
    
    if not notes_folder.exists():
        print(f"ERROR: Notes folder not found: {notes_folder}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize integrator
    integrator = PowerPointNotesIntegrator()
    
    # Create enhanced flashcards
    result = integrator.create_comprehensive_flashcards(powerpoint_path, notes_folder, output_dir)
    
    if result:
        print(f"\n" + "=" * 60)
        print("INTEGRATION COMPLETE")
        print("=" * 60)
        print(f"  PowerPoint: {powerpoint_path.name}")
        print(f"  Notes: {notes_folder.name}")
        print(f"  Lesson: Perform Effectively In An Operational Environment")
        print(f"  Original flashcards: {result['original_flashcards']}")
        print(f"  Enhanced flashcards: {result['enhanced_flashcards']}")
        print(f"  Notes insights integrated: {result['notes_insights']['total_words']} words")
        print(f"  Ready for Quizlet import!")

if __name__ == "__main__":
    main()
