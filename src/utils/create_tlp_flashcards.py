#!/usr/bin/env python3
"""
Create TLP Flashcards from Audio Transcription

Generates high-quality Quizlet flashcards from the TLP (Troop Leading Procedures)
audio transcription for military training.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

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

class TLPFlashcardGenerator:
    def __init__(self):
        """Initialize the flashcard generator with OpenAI client."""
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
    
    def load_audio_transcription(self, transcription_file: Path) -> Dict[str, Any]:
        """Load audio transcription data."""
        try:
            with open(transcription_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✓ Loaded audio transcription: {data.get('processing_metadata', {}).get('word_count', 0)} words")
            return data
            
        except Exception as e:
            print(f"ERROR loading audio transcription: {e}", file=sys.stderr)
            return {}
    
    def generate_flashcards_from_audio(self, audio_data: Dict) -> List[Dict[str, str]]:
        """Generate flashcards from audio transcription."""
        if not audio_data:
            print("ERROR: No audio data available", file=sys.stderr)
            return []
        
        print("Generating flashcards from TLP audio transcription...")
        
        # Prepare context for flashcard generation
        audio_text = audio_data.get('enhanced_text', '')
        test_material = audio_data.get('test_material', [])
        key_points = audio_data.get('key_points', [])
        military_acronyms = audio_data.get('military_acronyms_found', [])
        
        # Create flashcard generation prompt
        flashcard_prompt = f"""
        You are a military training expert creating high-quality flashcards for the Basic Officer Leader Course.
        
        AUDIO TRANSCRIPTION (TLP - Troop Leading Procedures):
        {audio_text[:4000]}  # Limit for API
        
        TEST MATERIAL SEGMENTS: {len(test_material)}
        KEY POINTS: {len(key_points)}
        MILITARY ACRONYMS FOUND: {', '.join(military_acronyms)}
        
        TASK: Create comprehensive Quizlet flashcards for TLP (Troop Leading Procedures) by:
        1. Identifying key terms, concepts, and definitions from the audio
        2. Creating clear, concise term-definition pairs
        3. Focusing on testable material and important concepts
        4. Including military acronyms and their definitions
        5. Emphasizing procedures, steps, and critical information
        6. Ensuring accuracy for military training context
        
        REQUIREMENTS:
        - Create 20-30 high-quality flashcards
        - Use clear, military-appropriate language
        - Focus on what would be on a test
        - Include both basic definitions and advanced concepts
        - Cover all major TLP topics mentioned in the audio
        
        Return ONLY a JSON array of flashcards with 'term' and 'definition' fields.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a military training assistant specializing in creating high-quality flashcards for test preparation. Focus on accuracy, clarity, and test-relevance."},
                    {"role": "user", "content": flashcard_prompt}
                ],
                temperature=0.3,
                max_tokens=2500
            )
            
            flashcard_content = response.choices[0].message.content
            
            # Try to parse the JSON response
            try:
                flashcards = json.loads(flashcard_content)
                print(f"✓ Generated {len(flashcards)} TLP flashcards")
                return flashcards
            except json.JSONDecodeError:
                print("ERROR: Could not parse generated flashcards", file=sys.stderr)
                return []
                
        except Exception as e:
            print(f"ERROR generating flashcards: {e}", file=sys.stderr)
            return []
    
    def create_tlp_flashcards(self, audio_file: Path, output_dir: Path) -> Dict[str, Any]:
        """Create TLP flashcards from audio transcription."""
        print("=" * 60)
        print("TLP FLASHCARD GENERATION FROM AUDIO")
        print("=" * 60)
        
        # Load audio transcription
        audio_stem = audio_file.stem
        transcription_file = Path(f"downloads/audio/processed/{audio_stem}_transcription.json")
        if not transcription_file.exists():
            print(f"ERROR: Audio transcription not found: {transcription_file}", file=sys.stderr)
            return {}
        
        audio_data = self.load_audio_transcription(transcription_file)
        
        # Generate flashcards from audio
        flashcards = self.generate_flashcards_from_audio(audio_data)
        
        if not flashcards:
            print("ERROR: No flashcards generated", file=sys.stderr)
            return {}
        
        # Create comprehensive result
        result = {
            "source_audio": audio_file.name,
            "lesson_topic": "Troop Leading Procedures (TLP)",
            "total_flashcards": len(flashcards),
            "audio_insights": {
                "word_count": audio_data.get('processing_metadata', {}).get('word_count', 0),
                "test_material_segments": len(audio_data.get('test_material', [])),
                "key_points": len(audio_data.get('key_points', [])),
                "military_acronyms": audio_data.get('military_acronyms_found', [])
            },
            "flashcards": flashcards,
            "processing_metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_used": "gpt-4o-mini",
                "source": "audio_transcription"
            }
        }
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive JSON
        json_file = output_dir / f"{audio_stem}_tlp_flashcards.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save Quizlet-ready TSV
        tsv_file = output_dir / f"{audio_stem}_tlp_flashcards.tsv"
        with open(tsv_file, 'w', encoding='utf-8') as f:
            f.write("term\tdefinition\n")
            for card in flashcards:
                term = card.get('term', '').replace('\t', ' ').replace('\n', ' ')
                definition = card.get('definition', '').replace('\t', ' ').replace('\n', ' ')
                f.write(f"{term}\t{definition}\n")
        
        # Save summary
        summary_file = output_dir / f"{audio_stem}_tlp_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("TLP FLASHCARDS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Audio source: {audio_file.name}\n")
            f.write(f"Lesson topic: Troop Leading Procedures (TLP)\n")
            f.write(f"Total flashcards: {len(flashcards)}\n")
            f.write(f"Audio word count: {result['audio_insights']['word_count']}\n")
            f.write(f"Test material segments: {result['audio_insights']['test_material_segments']}\n")
            f.write(f"Key points identified: {result['audio_insights']['key_points']}\n")
            f.write(f"Military acronyms: {', '.join(result['audio_insights']['military_acronyms'])}\n\n")
            
            f.write("GENERATED TLP FLASHCARDS:\n")
            f.write("-" * 30 + "\n")
            for i, card in enumerate(flashcards):
                f.write(f"{i+1}. {card.get('term', '')}\n")
                f.write(f"   {card.get('definition', '')}\n\n")
        
        print(f"✓ Results saved to: {output_dir}")
        print(f"  - {json_file.name}")
        print(f"  - {tsv_file.name}")
        print(f"  - {summary_file.name}")
        
        return result

def main():
    parser = argparse.ArgumentParser(description="Create TLP flashcards from audio transcription")
    parser.add_argument("audio_file", help="Path to audio file (.m4a, .mp3, .wav)")
    parser.add_argument("--output-dir", default="downloads/integrated", 
                       help="Output directory for TLP flashcards")
    
    args = parser.parse_args()
    
    audio_path = Path(args.audio_file)
    output_dir = Path(args.output_dir)
    
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize generator
    generator = TLPFlashcardGenerator()
    
    # Create TLP flashcards
    result = generator.create_tlp_flashcards(audio_path, output_dir)
    
    if result:
        print(f"\n" + "=" * 60)
        print("TLP FLASHCARD GENERATION COMPLETE")
        print("=" * 60)
        print(f"  Audio: {audio_path.name}")
        print(f"  Lesson: Troop Leading Procedures (TLP)")
        print(f"  Flashcards generated: {result['total_flashcards']}")
        print(f"  Audio insights integrated: {result['audio_insights']['word_count']} words")
        print(f"  Test material segments: {result['audio_insights']['test_material_segments']}")
        print(f"  Military acronyms: {len(result['audio_insights']['military_acronyms'])}")
        print(f"  Ready for Quizlet import!")

if __name__ == "__main__":
    main()




