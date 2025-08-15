#!/usr/bin/env python3
"""
PowerPoint + Notes Integration for Enhanced Quizlet Flashcards

Combines PowerPoint content with handwritten notes insights to create
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

class PowerPointNotesIntegrator:
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
    
    def enhance_flashcards_with_notes(self, flashcards: List[Dict], notes_data: Dict) -> List[Dict]:
        """Enhance PowerPoint flashcards with notes insights."""
        if not notes_data:
            print("WARNING: No notes data available, returning original flashcards")
            return flashcards
        
        print("Enhancing flashcards with notes insights...")
        
        # Prepare context for enhancement
        notes_text = ""
        for filename, result in notes_data.get('results', {}).items():
            notes_text += f"\n--- {filename} ---\n"
            notes_text += result.get('enhanced_text', '')
        
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
        
        TASK: Enhance these flashcards by:
        1. Adding missing key concepts from the handwritten notes
        2. Improving definitions with notes context
        3. Adding new flashcards for important notes-only content
        4. Ensuring military accuracy and test-focus
        5. Incorporating any additional insights from student notes
        
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
        
        # Enhance flashcards with notes
        enhanced_flashcards = self.enhance_flashcards_with_notes(flashcards, notes_data)
        
        # Create comprehensive result
        result = {
            "source_powerpoint": powerpoint_file.name,
            "source_notes": notes_folder.name,
            "lesson_topic": "Perform Effectively In An Operational Environment",
            "original_flashcards": len(flashcards),
            "enhanced_flashcards": len(enhanced_flashcards),
            "notes_insights": {
                "total_images": notes_data.get('total_images', 0),
                "average_confidence": notes_data.get('average_confidence', 0),
                "total_words": notes_data.get('total_words', 0)
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
