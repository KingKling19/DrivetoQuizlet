#!/usr/bin/env python3
"""
PowerPoint + Audio Integration for Enhanced Quizlet Flashcards

Combines PowerPoint content with audio transcription insights to create
high-quality, comprehensive flashcards for military training.
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

class PowerPointAudioIntegrator:
    def __init__(self):
        """Initialize the integrator with OpenAI client."""
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
    
    def load_powerpoint_flashcards(self, quizlet_file: Path) -> List[Dict[str, str]]:
        """Load existing PowerPoint flashcards from Quizlet export."""
        try:
            with open(quizlet_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            flashcards = []
            for card in data.get('cards', []):
                flashcards.append({
                    'term': card.get('term', ''),
                    'definition': card.get('definition', '')
                })
            
            print(f"✓ Loaded {len(flashcards)} PowerPoint flashcards")
            return flashcards
            
        except Exception as e:
            print(f"ERROR loading PowerPoint flashcards: {e}", file=sys.stderr)
            return []
    
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
    
    def enhance_flashcards_with_audio(self, flashcards: List[Dict], audio_data: Dict) -> List[Dict]:
        """Enhance PowerPoint flashcards with audio insights."""
        if not audio_data:
            print("WARNING: No audio data available, returning original flashcards")
            return flashcards
        
        print("Enhancing flashcards with audio insights...")
        
        # Prepare context for enhancement
        audio_text = audio_data.get('enhanced_text', '')
        test_material = audio_data.get('test_material', [])
        key_points = audio_data.get('key_points', [])
        military_acronyms = audio_data.get('military_acronyms_found', [])
        
        # Create enhancement prompt
        enhancement_prompt = f"""
        You are a military training expert. I have PowerPoint flashcards and audio transcription from a lecture.
        
        AUDIO TRANSCRIPTION (TLP - Troop Leading Procedures):
        {audio_text[:3000]}  # Limit for API
        
        TEST MATERIAL SEGMENTS: {len(test_material)}
        KEY POINTS: {len(key_points)}
        MILITARY ACRONYMS FOUND: {', '.join(military_acronyms)}
        
        EXISTING POWERPOINT FLASHCARDS:
        {json.dumps(flashcards[:10], indent=2)}  # Show first 10 for context
        
        TASK: Enhance these flashcards by:
        1. Adding missing key concepts from the audio
        2. Improving definitions with audio context
        3. Adding new flashcards for important audio-only content
        4. Ensuring military accuracy and test-focus
        
        Return ONLY a JSON array of enhanced flashcards with 'term' and 'definition' fields.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a military training assistant specializing in creating high-quality flashcards for test preparation."},
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
    
    def create_comprehensive_flashcards(self, powerpoint_file: Path, audio_file: Path, output_dir: Path) -> Dict[str, Any]:
        """Create comprehensive flashcards combining PowerPoint and audio."""
        print("=" * 60)
        print("POWERPOINT + AUDIO FLASHCARD INTEGRATION")
        print("=" * 60)
        
        # Load PowerPoint flashcards
        quizlet_file = powerpoint_file.with_suffix('.quizlet.json')
        if not quizlet_file.exists():
            print(f"ERROR: Quizlet file not found: {quizlet_file}", file=sys.stderr)
            return {}
        
        flashcards = self.load_powerpoint_flashcards(quizlet_file)
        if not flashcards:
            print("ERROR: No flashcards loaded", file=sys.stderr)
            return {}
        
        # Load audio transcription
        audio_stem = audio_file.stem
        transcription_file = Path(f"downloads/audio/processed/{audio_stem}_transcription.json")
        if not transcription_file.exists():
            print(f"ERROR: Audio transcription not found: {transcription_file}", file=sys.stderr)
            return {}
        
        audio_data = self.load_audio_transcription(transcription_file)
        
        # Enhance flashcards with audio
        enhanced_flashcards = self.enhance_flashcards_with_audio(flashcards, audio_data)
        
        # Create comprehensive result
        result = {
            "source_powerpoint": powerpoint_file.name,
            "source_audio": audio_file.name,
            "original_flashcards": len(flashcards),
            "enhanced_flashcards": len(enhanced_flashcards),
            "audio_insights": {
                "word_count": audio_data.get('processing_metadata', {}).get('word_count', 0),
                "test_material_segments": len(audio_data.get('test_material', [])),
                "key_points": len(audio_data.get('key_points', [])),
                "military_acronyms": audio_data.get('military_acronyms_found', [])
            },
            "flashcards": enhanced_flashcards,
            "processing_metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_used": "gpt-4o-mini"
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
            f.write(f"Audio source: {audio_file.name}\n")
            f.write(f"Original flashcards: {len(flashcards)}\n")
            f.write(f"Enhanced flashcards: {len(enhanced_flashcards)}\n")
            f.write(f"Audio word count: {result['audio_insights']['word_count']}\n")
            f.write(f"Test material segments: {result['audio_insights']['test_material_segments']}\n")
            f.write(f"Key points identified: {result['audio_insights']['key_points']}\n")
            f.write(f"Military acronyms: {', '.join(result['audio_insights']['military_acronyms'])}\n\n")
            
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
    parser = argparse.ArgumentParser(description="Integrate PowerPoint and audio for enhanced flashcards")
    parser.add_argument("powerpoint_file", help="Path to PowerPoint file (.pptx)")
    parser.add_argument("audio_file", help="Path to audio file (.m4a, .mp3, .wav)")
    parser.add_argument("--output-dir", default="downloads/integrated", 
                       help="Output directory for integrated files")
    
    args = parser.parse_args()
    
    powerpoint_path = Path(args.powerpoint_file)
    audio_path = Path(args.audio_file)
    output_dir = Path(args.output_dir)
    
    if not powerpoint_path.exists():
        print(f"ERROR: PowerPoint file not found: {powerpoint_path}", file=sys.stderr)
        sys.exit(1)
    
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize integrator
    integrator = PowerPointAudioIntegrator()
    
    # Create enhanced flashcards
    result = integrator.create_comprehensive_flashcards(powerpoint_path, audio_path, output_dir)
    
    if result:
        print(f"\n" + "=" * 60)
        print("INTEGRATION COMPLETE")
        print("=" * 60)
        print(f"  PowerPoint: {powerpoint_path.name}")
        print(f"  Audio: {audio_path.name}")
        print(f"  Original flashcards: {result['original_flashcards']}")
        print(f"  Enhanced flashcards: {result['enhanced_flashcards']}")
        print(f"  Audio insights integrated: {result['audio_insights']['word_count']} words")
        print(f"  Test material segments: {result['audio_insights']['test_material_segments']}")
        print(f"  Ready for Quizlet import!")

if __name__ == "__main__":
    main()




