#!/usr/bin/env python3
"""
Notes Processor for Military Training Materials

Extracts text from handwritten notes (PNG files) and integrates them
with PowerPoint and audio content to create comprehensive study materials.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. pip install Pillow", file=sys.stderr)
    sys.exit(1)

try:
    import easyocr
except ImportError:
    print("WARNING: easyocr not installed. Install with: pip install easyocr", file=sys.stderr)
    easyocr = None

try:
    from openai import OpenAI
except ImportError:
    print("WARNING: openai package not found. Summary generation will be limited.", file=sys.stderr)
    OpenAI = None

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env file")
except ImportError:
    print("WARNING: python-dotenv not installed. Install with: pip install python-dotenv", file=sys.stderr)
except Exception as e:
    print(f"WARNING: Could not load .env file: {e}", file=sys.stderr)

# Military context for better interpretation
MILITARY_ACRONYMS = {
    "TLP": "Troop Leading Procedures",
    "OPORD": "Operations Order",
    "FRAGO": "Fragmentary Order",
    "SOP": "Standard Operating Procedure",
    "METT-TC": "Mission, Enemy, Terrain, Troops, Time, Civilian considerations",
    "OAKOC": "Observation and fields of fire, Avenues of approach, Key terrain, Obstacles, Cover and concealment",
    "COA": "Course of Action",
    "WARNO": "Warning Order",
    "CONOP": "Concept of Operations",
    "SITREP": "Situation Report",
    "ACE": "Ammunition, Casualties, Equipment",
    "SALUTE": "Size, Activity, Location, Unit, Time, Equipment",
    "AD": "Air Defense",
    "ADA": "Air Defense Artillery",
    "BOLC": "Basic Officer Leader Course",
    "MDO": "Multi-Domain Operations",
    "LSCO": "Large Scale Combat Operations",
    "CBRNE": "Chemical, Biological, Radiological, Nuclear, Explosive",
    "MANPADS": "Man-Portable Air Defense System",
    "IED": "Improvised Explosive Device",
    "C2": "Command and Control",
    "EW": "Electronic Warfare",
    "PSO": "Private Security Organization",
    "OE": "Operational Environment",
    "PMESII-PT": "Political, Military, Economic, Social, Information, Infrastructure, Physical Environment, Time",
    "SCU": "Socio-Cultural Understanding",
    "VBBN": "Values, Beliefs, Behaviors, and Norms"
}

class NotesProcessor:
    def __init__(self, use_gpu: bool = True):
        """Initialize the notes processor with lazy loading."""
        self.use_gpu = use_gpu
        self.reader = None
        self.client = None
        
        # Note: Models will be loaded lazily when first needed
        print("✓ NotesProcessor initialized (models will load on first use)")
    
    def extract_text_from_image(self, image_path: Path) -> Dict[str, Any]:
        """Extract text from a handwritten note image."""
        # Lazy load OCR reader only when needed
        if self.reader is None:
            try:
                from model_manager import model_manager
                self.reader = model_manager.get_ocr_reader(['en'], self.use_gpu)
            except ImportError:
                # Fallback to direct loading if model manager not available
                if easyocr:
                    try:
                        print("Initializing OCR reader...")
                        self.reader = easyocr.Reader(['en'], gpu=self.use_gpu)
                        print("✓ OCR reader initialized")
                    except Exception as e:
                        print(f"WARNING: Could not initialize OCR reader: {e}", file=sys.stderr)
                        return {"text": "", "confidence": 0.0, "error": f"OCR not available: {e}"}
                else:
                    print("ERROR: OCR reader not available", file=sys.stderr)
                    return {"text": "", "confidence": 0.0, "error": "OCR not available"}
        
        try:
            print(f"Processing image: {image_path.name}")
            
            # Read image
            image = Image.open(image_path)
            
            # Extract text with OCR
            results = self.reader.readtext(str(image_path))
            
            # Process results
            extracted_text = []
            total_confidence = 0.0
            valid_results = 0
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Filter low confidence results
                    extracted_text.append(text.strip())
                    total_confidence += confidence
                    valid_results += 1
            
            # Calculate average confidence
            avg_confidence = total_confidence / valid_results if valid_results > 0 else 0.0
            
            # Combine text
            full_text = " ".join(extracted_text)
            
            # Enhance with military context
            enhanced_text = self._enhance_text_with_context(full_text)
            
            return {
                "original_text": full_text,
                "enhanced_text": enhanced_text,
                "confidence": avg_confidence,
                "word_count": len(full_text.split()),
                "valid_segments": valid_results,
                "total_segments": len(results)
            }
            
        except Exception as e:
            print(f"ERROR processing {image_path.name}: {e}", file=sys.stderr)
            return {"text": "", "confidence": 0.0, "error": str(e)}
    
    def _enhance_text_with_context(self, text: str) -> str:
        """Enhance extracted text with military acronym context."""
        enhanced = text
        
        # Replace common OCR errors with military terms
        replacements = {
            "t l p": "TLP",
            "t l p's": "TLPs",
            "o p o r d": "OPORD",
            "f r a g o": "FRAGO",
            "s o p": "SOP",
            "m e t t t c": "METT-TC",
            "o a k o c": "OAKOC",
            "c o a": "COA",
            "w a r n o": "WARNO",
            "c o n o p": "CONOP",
            "s i t r e p": "SITREP",
            "a c e": "ACE",
            "s a l u t e": "SALUTE",
            "a d": "AD",
            "a d a": "ADA",
            "b o l c": "BOLC",
            "m d o": "MDO",
            "l s c o": "LSCO",
            "c b r n e": "CBRNE",
            "m a n p a d s": "MANPADS",
            "i e d": "IED",
            "i e d's": "IEDs",
            "c 2": "C2",
            "e w": "EW",
            "p s o": "PSO",
            "o e": "OE",
            "p m e s i i p t": "PMESII-PT",
            "s c u": "SCU",
            "v b b n": "VBBN"
        }
        
        for wrong, correct in replacements.items():
            enhanced = re.sub(rf'\b{wrong}\b', correct, enhanced, flags=re.IGNORECASE)
        
        return enhanced
    
    def process_notes_folder(self, notes_folder: Path, output_dir: Path) -> Dict[str, Any]:
        """Process all note images in a folder."""
        if not notes_folder.exists():
            print(f"ERROR: Notes folder not found: {notes_folder}", file=sys.stderr)
            return {}
        
        # Find all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(notes_folder.glob(f"*{ext}"))
            image_files.extend(notes_folder.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No image files found in {notes_folder}")
            return {}
        
        print(f"Found {len(image_files)} image files")
        
        # Process each image
        results = {}
        total_confidence = 0.0
        total_words = 0
        
        for image_file in sorted(image_files):
            result = self.extract_text_from_image(image_file)
            results[image_file.name] = result
            
            if result.get("confidence", 0) > 0:
                total_confidence += result["confidence"]
                total_words += result.get("word_count", 0)
        
        # Calculate overall statistics
        avg_confidence = total_confidence / len(results) if results else 0.0
        
        # Create summary
        summary = {
            "notes_folder": str(notes_folder),
            "total_images": len(image_files),
            "processed_images": len(results),
            "average_confidence": avg_confidence,
            "total_words": total_words,
            "results": results,
            "processing_metadata": {
                "timestamp": datetime.now().isoformat(),
                "ocr_engine": "easyocr",
                "gpu_used": self.use_gpu
            }
        }
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_file = output_dir / f"{notes_folder.name}_notes_processed.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save text summary
        text_file = output_dir / f"{notes_folder.name}_notes_processed.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("HANDWRITTEN NOTES PROCESSING RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Notes folder: {notes_folder.name}\n")
            f.write(f"Total images: {len(image_files)}\n")
            f.write(f"Average confidence: {avg_confidence:.2f}\n")
            f.write(f"Total words extracted: {total_words}\n\n")
            
            for filename, result in results.items():
                f.write(f"--- {filename} ---\n")
                f.write(f"Confidence: {result.get('confidence', 0):.2f}\n")
                f.write(f"Words: {result.get('word_count', 0)}\n")
                f.write("Enhanced text:\n")
                f.write(result.get('enhanced_text', '') + "\n\n")
        
        print(f"✓ Results saved to: {output_dir}")
        print(f"  - {json_file.name}")
        print(f"  - {text_file.name}")
        
        return summary
    
    def generate_ai_analysis(self, notes_summary: Dict, output_dir: Path) -> Optional[str]:
        """Generate AI analysis of the handwritten notes."""
        if not self.client:
            print("WARNING: OpenAI client not available. Skipping AI analysis.")
            return None
        
        try:
            print("Generating AI analysis of notes...")
            
            # Prepare context
            context = f"""
            This is a collection of handwritten notes from military training, specifically for the Basic Officer Leader Course for Air Defense Artillery.
            
            Key context:
            - Focus on testable material and important concepts
            - Military acronyms and terminology are common
            - Look for definitions, procedures, and key principles
            - Emphasize anything marked as important or test material
            
            Processing statistics:
            - Total images processed: {notes_summary['total_images']}
            - Average confidence: {notes_summary['average_confidence']:.2f}
            - Total words extracted: {notes_summary['total_words']}
            
            EXTRACTED NOTES:
            """
            
            # Add extracted text (limit to avoid token limits)
            for filename, result in notes_summary['results'].items():
                context += f"\n--- {filename} ---\n"
                context += result.get('enhanced_text', '')[:500] + "\n"
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a military training assistant. Analyze these handwritten notes and create a comprehensive summary focusing on testable material, key concepts, and important definitions. Use bullet points and organize by topic."},
                    {"role": "user", "content": context}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            analysis = response.choices[0].message.content
            
            # Save analysis
            analysis_file = output_dir / f"{Path(notes_summary['notes_folder']).name}_notes_analysis.txt"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write("HANDWRITTEN NOTES AI ANALYSIS\n")
                f.write("=" * 50 + "\n")
                f.write(analysis)
            
            print(f"✓ AI Analysis saved to: {analysis_file.name}")
            return analysis
            
        except Exception as e:
            print(f"ERROR generating AI analysis: {e}", file=sys.stderr)
            return None

def main():
    parser = argparse.ArgumentParser(description="Process handwritten notes with OCR and AI analysis")
    parser.add_argument("notes_folder", help="Path to folder containing note images")
    parser.add_argument("--output-dir", default="downloads/notes/processed", 
                       help="Output directory for processed files")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU usage for OCR")
    parser.add_argument("--generate-analysis", action="store_true", help="Generate AI analysis")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    notes_folder = Path(args.notes_folder)
    if not notes_folder.exists():
        print(f"ERROR: Notes folder not found: {notes_folder}", file=sys.stderr)
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("HANDWRITTEN NOTES PROCESSOR")
    print("=" * 60)
    
    # Initialize processor
    processor = NotesProcessor(use_gpu=not args.no_gpu)
    
    # Process notes
    summary = processor.process_notes_folder(notes_folder, output_dir)
    
    # Generate AI analysis if requested
    if args.generate_analysis and summary:
        processor.generate_ai_analysis(summary, output_dir)
    
    # Print summary statistics
    if summary:
        print(f"\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"  Notes folder: {notes_folder.name}")
        print(f"  Total images: {summary['total_images']}")
        print(f"  Average confidence: {summary['average_confidence']:.2f}")
        print(f"  Total words extracted: {summary['total_words']}")

if __name__ == "__main__":
    main()




