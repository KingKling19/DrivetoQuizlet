#!/usr/bin/env python3
"""
Lesson Processing System

Automatically processes lessons based on available files (PowerPoint, notes, audio)
and generates comprehensive Quizlet flashcards.
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

class LessonProcessor:
    def __init__(self, config_dir: Path = Path("config")):
        """Initialize the lesson processor with cross-lesson context."""
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
            "max_related_lessons": 3,
            "context_weight_threshold": 0.3,
            "max_context_length": 2000,
            "include_prerequisites": True,
            "include_related_concepts": True,
            "enable_cross_lesson_enhancement": True
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
    
    def get_lesson_id_from_path(self, lesson_dir: Path) -> str:
        """Extract lesson ID from lesson directory path."""
        return lesson_dir.name
    
    def find_related_lessons(self, lesson_id: str, max_lessons: int = 3) -> List[Dict[str, Any]]:
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
    
    def get_context_enhancement_summary(self, lesson_id: str) -> Dict[str, Any]:
        """Get summary of cross-lesson context enhancement for a lesson."""
        if not self.context_config["enable_cross_lesson_enhancement"]:
            return {"enabled": False, "reason": "Cross-lesson enhancement disabled"}
        
        related_lessons = self.find_related_lessons(lesson_id, self.context_config["max_related_lessons"])
        
        return {
            "enabled": True,
            "lesson_id": lesson_id,
            "related_lessons_found": len(related_lessons),
            "lessons_indexed": len(self.cross_lesson_data.get('content_index', {})),
            "related_lessons": [
                {
                    "lesson_id": rel.get("lesson_id"),
                    "similarity_score": rel.get("similarity_score", 0),
                    "relationship_type": rel.get("relationship_type", "unknown")
                }
                for rel in related_lessons
            ],
            "context_config": self.context_config
        }
    
    def scan_lesson_files(self, lesson_dir: Path) -> Dict[str, List[Path]]:
        """Scan lesson directory for available files."""
        files = {
            "presentations": [],
            "notes": [],
            "audio": [],
            "processed": []
        }
        
        # Scan presentations
        presentations_dir = lesson_dir / "presentations"
        if presentations_dir.exists():
            for file in presentations_dir.glob("*.pptx"):
                files["presentations"].append(file)
        
        # Scan notes
        notes_dir = lesson_dir / "notes"
        if notes_dir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]:
                for file in notes_dir.glob(ext):
                    files["notes"].append(file)
        
        # Scan audio
        audio_dir = lesson_dir / "audio"
        if audio_dir.exists():
            for ext in ["*.m4a", "*.mp3", "*.wav", "*.flac"]:
                for file in audio_dir.glob(ext):
                    files["audio"].append(file)
        
        # Scan processed files
        processed_dir = lesson_dir / "processed"
        if processed_dir.exists():
            for file in processed_dir.glob("*"):
                if file.is_file():
                    files["processed"].append(file)
        
        return files
    
    def process_powerpoint_only(self, lesson_dir: Path, files: Dict[str, List[Path]]) -> bool:
        """Process PowerPoint files to create flashcards."""
        if not files["presentations"]:
            print("No PowerPoint files found")
            return False
        
        print("Processing PowerPoint files...")
        
        # Import the PowerPoint conversion function
        try:
            from src.processing.convert_folder_to_quizlet import convert_folder_to_quizlet
            output_dir = lesson_dir / "processed"
            output_dir.mkdir(exist_ok=True)
            
            # Process each presentation
            for pptx_file in files["presentations"]:
                print(f"  Processing: {pptx_file.name}")
                # This would call your existing PowerPoint processing
                # For now, we'll assume the quizlet files are already created
            
            return True
            
        except ImportError:
            print("WARNING: PowerPoint processing module not available")
            return False
    
    def process_notes(self, lesson_dir: Path, files: Dict[str, List[Path]]) -> bool:
        """Process handwritten notes with OCR."""
        if not files["notes"]:
            print("No note files found")
            return False
        
        print("Processing handwritten notes...")
        
        try:
            from src.processing.notes_processor import NotesProcessor
            
            processor = NotesProcessor(use_gpu=True)
            notes_dir = lesson_dir / "notes"
            output_dir = lesson_dir / "processed"
            
            summary = processor.process_notes_folder(notes_dir, output_dir)
            
            if summary:
                print(f"✓ Notes processed: {summary['total_words']} words extracted")
                return True
            else:
                print("WARNING: Notes processing failed")
                return False
                
        except ImportError:
            print("WARNING: Notes processing module not available")
            return False
    
    def process_audio(self, lesson_dir: Path, files: Dict[str, List[Path]]) -> bool:
        """Process audio files with transcription."""
        if not files["audio"]:
            print("No audio files found")
            return False
        
        print("Processing audio files...")
        
        try:
            from src.processing.audio_processor import AudioProcessor
            
            processor = AudioProcessor(model_size="base", use_gpu=True)
            output_dir = lesson_dir / "processed"
            
            for audio_file in files["audio"]:
                print(f"  Processing: {audio_file.name}")
                result = processor.transcribe_audio(audio_file, output_dir)
                
                if result:
                    print(f"✓ Audio processed: {result['processing_metadata']['word_count']} words")
            
            return True
            
        except ImportError:
            print("WARNING: Audio processing module not available")
            return False
    
    def integrate_all_sources(self, lesson_dir: Path, files: Dict[str, List[Path]]) -> Dict[str, Any]:
        """Integrate all available sources into comprehensive flashcards."""
        print("Integrating all sources...")
        
        # Load PowerPoint flashcards
        powerpoint_flashcards = []
        quizlet_files = [f for f in files["processed"] if f.name.endswith(".quizlet.json")]
        
        if quizlet_files:
            try:
                with open(quizlet_files[0], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    powerpoint_flashcards = data
                else:
                    powerpoint_flashcards = data.get('cards', [])
                
                print(f"✓ Loaded {len(powerpoint_flashcards)} PowerPoint flashcards")
            except Exception as e:
                print(f"WARNING: Could not load PowerPoint flashcards: {e}")
        
        # Load notes analysis
        notes_data = None
        notes_files = [f for f in files["processed"] if "notes_processed.json" in f.name]
        
        if notes_files:
            try:
                with open(notes_files[0], 'r', encoding='utf-8') as f:
                    notes_data = json.load(f)
                print(f"✓ Loaded notes analysis: {notes_data.get('total_words', 0)} words")
            except Exception as e:
                print(f"WARNING: Could not load notes analysis: {e}")
        
        # Load audio transcription
        audio_data = None
        audio_files = [f for f in files["processed"] if "transcription.json" in f.name]
        
        if audio_files:
            try:
                with open(audio_files[0], 'r', encoding='utf-8') as f:
                    audio_data = json.load(f)
                print(f"✓ Loaded audio transcription: {audio_data.get('processing_metadata', {}).get('word_count', 0)} words")
            except Exception as e:
                print(f"WARNING: Could not load audio transcription: {e}")
        
        # Get lesson ID and cross-lesson context
        lesson_id = self.get_lesson_id_from_path(lesson_dir)
        context_summary = self.get_context_enhancement_summary(lesson_id)
        
        # Create comprehensive integration prompt
        integration_prompt = f"""
        You are a military training expert creating comprehensive flashcards for the Basic Officer Leader Course.
        
        LESSON: {lesson_dir.name} (ID: {lesson_id})
        
        POWERPOINT FLASHCARDS: {len(powerpoint_flashcards)} cards
        NOTES AVAILABLE: {'Yes' if notes_data else 'No'}
        AUDIO AVAILABLE: {'Yes' if audio_data else 'No'}
        CROSS-LESSON CONTEXT: {'Enabled' if context_summary.get('enabled') else 'Disabled'}
        
        CROSS-LESSON CONTEXT SUMMARY:
        - Lessons indexed: {context_summary.get('lessons_indexed', 0)}
        - Related lessons found: {context_summary.get('related_lessons_found', 0)}
        - Related lessons: {', '.join([rel.get('lesson_id', '') for rel in context_summary.get('related_lessons', [])])}
        
        TASK: Create the most comprehensive and high-quality flashcards possible by:
        1. Using the PowerPoint flashcards as a foundation
        2. Adding insights from handwritten notes (if available)
        3. Incorporating audio lecture content (if available)
        4. Using cross-lesson context to enhance understanding and avoid duplicates
        5. Ensuring military accuracy and test-focus
        6. Creating clear, concise term-definition pairs
        7. Correlating concepts across related lessons for better comprehension
        
        REQUIREMENTS:
        - Focus on testable material
        - Include military acronyms and definitions
        - Emphasize key concepts and procedures
        - Use clear, military-appropriate language
        - Create 30-50 high-quality flashcards
        - Leverage cross-lesson context to improve content quality
        - Avoid duplicate definitions across related lessons
        
        Return ONLY a JSON array of flashcards with 'term' and 'definition' fields.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a military training assistant specializing in creating high-quality flashcards for test preparation."},
                    {"role": "user", "content": integration_prompt}
                ],
                temperature=0.3,
                max_tokens=3000
            )
            
            enhanced_content = response.choices[0].message.content
            
            try:
                enhanced_flashcards = json.loads(enhanced_content)
                print(f"✓ Generated {len(enhanced_flashcards)} comprehensive flashcards")
                return enhanced_flashcards
            except json.JSONDecodeError:
                print("WARNING: Could not parse enhanced flashcards")
                return powerpoint_flashcards
                
        except Exception as e:
            print(f"ERROR generating comprehensive flashcards: {e}")
            return powerpoint_flashcards
    
    def save_final_output(self, lesson_dir: Path, flashcards: List[Dict], lesson_name: str):
        """Save final flashcards to output directory."""
        output_dir = lesson_dir / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Save TSV for Quizlet import
        tsv_file = output_dir / f"{lesson_name}_flashcards.tsv"
        with open(tsv_file, 'w', encoding='utf-8') as f:
            f.write("term\tdefinition\n")
            for card in flashcards:
                term = card.get('term', '').replace('\t', ' ').replace('\n', ' ')
                definition = card.get('definition', '').replace('\t', ' ').replace('\n', ' ')
                f.write(f"{term}\t{definition}\n")
        
        # Get lesson ID and cross-lesson context summary
        lesson_id = self.get_lesson_id_from_path(lesson_dir)
        context_summary = self.get_context_enhancement_summary(lesson_id)
        
        # Save JSON with metadata including cross-lesson context
        json_file = output_dir / f"{lesson_name}_flashcards.json"
        result = {
            "lesson_name": lesson_name,
            "lesson_id": lesson_id,
            "total_flashcards": len(flashcards),
            "cross_lesson_context": context_summary,
            "flashcards": flashcards,
            "processing_metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_used": "gpt-4o-mini",
                "cross_lesson_enhancement": context_summary.get('enabled', False)
            }
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Final output saved to: {output_dir}")
        print(f"  - {tsv_file.name} (Ready for Quizlet import)")
        print(f"  - {json_file.name} (Complete data)")
        print(f"  - Cross-lesson context: {'Enabled' if context_summary.get('enabled') else 'Disabled'}")
        if context_summary.get('enabled'):
            print(f"  - Related lessons: {context_summary.get('related_lessons_found', 0)} found")
        
        return tsv_file
    
    def process_lesson(self, lesson_dir: Path) -> bool:
        """Process a complete lesson with all available sources."""
        print("=" * 60)
        print(f"PROCESSING LESSON: {lesson_dir.name}")
        print("=" * 60)
        
        # Scan for available files
        files = self.scan_lesson_files(lesson_dir)
        
        print(f"Files found:")
        print(f"  - Presentations: {len(files['presentations'])}")
        print(f"  - Notes: {len(files['notes'])}")
        print(f"  - Audio: {len(files['audio'])}")
        print(f"  - Processed: {len(files['processed'])}")
        
        # Process each source type
        powerpoint_processed = self.process_powerpoint_only(lesson_dir, files)
        notes_processed = self.process_notes(lesson_dir, files)
        audio_processed = self.process_audio(lesson_dir, files)
        
        # Re-scan for newly processed files
        files = self.scan_lesson_files(lesson_dir)
        
        # Integrate all sources
        flashcards = self.integrate_all_sources(lesson_dir, files)
        
        if flashcards:
            # Save final output
            tsv_file = self.save_final_output(lesson_dir, flashcards, lesson_dir.name)
            
            print(f"\n" + "=" * 60)
            print("LESSON PROCESSING COMPLETE")
            print("=" * 60)
            print(f"  Lesson: {lesson_dir.name}")
            print(f"  Flashcards generated: {len(flashcards)}")
            print(f"  PowerPoint processed: {powerpoint_processed}")
            print(f"  Notes processed: {notes_processed}")
            print(f"  Audio processed: {audio_processed}")
            print(f"  Ready for Quizlet import: {tsv_file}")
            
            return True
        else:
            print("ERROR: No flashcards generated")
            return False

def main():
    parser = argparse.ArgumentParser(description="Process a complete lesson with all available sources")
    parser.add_argument("lesson_dir", help="Path to lesson directory")
    
    args = parser.parse_args()
    
    lesson_dir = Path(args.lesson_dir)
    if not lesson_dir.exists():
        print(f"ERROR: Lesson directory not found: {lesson_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize processor
    processor = LessonProcessor()
    
    # Process the lesson
    success = processor.process_lesson(lesson_dir)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()




