#!/usr/bin/env python3
"""
Audio Processor for Military Training Lectures

Transcribes audio files with military context awareness and generates
test-focused summaries highlighting key points and testable material.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded environment variables from .env file")
except ImportError:
    print("WARNING: python-dotenv not installed. Install with: pip install python-dotenv", file=sys.stderr)
except Exception as e:
    print(f"WARNING: Could not load .env file: {e}", file=sys.stderr)

try:
    import whisper
except ImportError:
    print("ERROR: openai-whisper is required. pip install openai-whisper", file=sys.stderr)
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("WARNING: openai package not found. Summary generation will be limited.", file=sys.stderr)
    OpenAI = None

# Military context for better transcription
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

# Test material indicators
TEST_INDICATORS = [
    "this will be on the test",
    "this is important",
    "remember this",
    "key point",
    "critical",
    "essential",
    "must know",
    "testable material",
    "you need to know",
    "important to remember",
    "this is critical",
    "pay attention to",
    "this is key",
    "this is essential",
    "this is fundamental"
]

class AudioProcessor:
    def __init__(self, model_size: str = "base", use_gpu: bool = True):
        """Initialize the audio processor with lazy loading."""
        self.model_size = model_size
        self.use_gpu = use_gpu
        self.model = None
        self.client = None
        
        # Check GPU availability
        self._check_gpu_status()
        
        # Note: Models will be loaded lazily when first needed
        print("✓ AudioProcessor initialized (models will load on first use)")
    
    def _check_gpu_status(self):
        """Check and report GPU status."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                print(f"✓ GPU available: {gpu_name} (CUDA: {torch.version.cuda})")
                if not self.use_gpu:
                    print("WARNING: GPU available but --no-gpu flag used")
            else:
                print("⚠️  GPU not available, will use CPU")
                if self.use_gpu:
                    print("WARNING: GPU requested but not available, falling back to CPU")
        except ImportError:
            print("⚠️  PyTorch not available, cannot check GPU status")
    
    def load_model(self):
        """Load the Whisper model using the model manager."""
        if self.model is None:
            try:
                from model_manager import model_manager
                self.model = model_manager.get_whisper_model(self.model_size, self.use_gpu)
            except ImportError:
                # Fallback to direct loading if model manager not available
                print(f"Loading Whisper model: {self.model_size}")
                try:
                    import whisper
                    self.model = whisper.load_model(self.model_size)
                    device = "GPU" if self.model.device.type == "cuda" else "CPU"
                    print(f"✓ Model loaded on {device}: {self.model.device}")
                except Exception as e:
                    print(f"ERROR loading model: {e}", file=sys.stderr)
                    raise
    
    def transcribe_audio(self, audio_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Transcribe audio file with military context awareness."""
        # Lazy load model only when needed
        if self.model is None:
            self.load_model()
        
        print(f"Transcribing: {audio_path.name}")
        start_time = time.time()
        
        try:
            # Transcribe with military context
            result = self.model.transcribe(
                str(audio_path),
                language="en",
                word_timestamps=True,
                verbose=True
            )
            
            transcription_time = time.time() - start_time
            print(f"✓ Transcription completed in {transcription_time:.2f} seconds")
            
            # Process transcription with military context
            processed_result = self._process_transcription(result, audio_path)
            
            # Save results
            self._save_transcription(processed_result, output_dir, audio_path.stem)
            
            return processed_result
            
        except Exception as e:
            print(f"ERROR during transcription: {e}", file=sys.stderr)
            raise
    
    def _process_transcription(self, result: Dict, audio_path: Path) -> Dict[str, Any]:
        """Process transcription with military context and test material detection."""
        text = result.get("text", "")
        segments = result.get("segments", [])
        
        # Clean and enhance text
        enhanced_text = self._enhance_text_with_context(text)
        
        # Detect test material
        test_material = self._detect_test_material(segments)
        
        # Extract key points
        key_points = self._extract_key_points(segments)
        
        return {
            "audio_file": audio_path.name,
            "duration": result.get("language", "en"),
            "language": result.get("language", "en"),
            "enhanced_text": enhanced_text,
            "original_text": text,
            "segments": segments,
            "test_material": test_material,
            "key_points": key_points,
            "military_acronyms_found": self._find_military_acronyms(text),
            "processing_metadata": {
                "model_used": self.model_size,
                "timestamp": datetime.now().isoformat(),
                "word_count": len(text.split()),
                "segment_count": len(segments)
            }
        }
    
    def _enhance_text_with_context(self, text: str) -> str:
        """Enhance transcription with military acronym context."""
        enhanced = text
        
        # Replace common transcription errors with military terms
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
    
    def _detect_test_material(self, segments: List[Dict]) -> List[Dict]:
        """Detect segments that contain test material indicators."""
        test_segments = []
        
        for segment in segments:
            text = segment.get("text", "").lower()
            
            # Check for test indicators
            for indicator in TEST_INDICATORS:
                if indicator in text:
                    test_segments.append({
                        "start": segment.get("start", 0),
                        "end": segment.get("end", 0),
                        "text": segment.get("text", ""),
                        "indicator_found": indicator,
                        "confidence": segment.get("avg_logprob", 0)
                    })
                    break
        
        return test_segments
    
    def _extract_key_points(self, segments: List[Dict]) -> List[Dict]:
        """Extract key points based on content analysis."""
        key_points = []
        
        for segment in segments:
            text = segment.get("text", "")
            
            # Look for patterns that indicate key information
            if any(pattern in text.lower() for pattern in [
                "definition", "means", "refers to", "is defined as",
                "consists of", "includes", "comprises", "contains",
                "primary", "main", "key", "essential", "critical",
                "first", "second", "third", "finally", "lastly"
            ]):
                key_points.append({
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": text,
                    "confidence": segment.get("avg_logprob", 0)
                })
        
        return key_points
    
    def _find_military_acronyms(self, text: str) -> List[str]:
        """Find military acronyms mentioned in the text."""
        found_acronyms = []
        text_upper = text.upper()
        
        for acronym in MILITARY_ACRONYMS.keys():
            if acronym in text_upper:
                found_acronyms.append(acronym)
        
        return found_acronyms
    
    def _save_transcription(self, result: Dict, output_dir: Path, filename: str):
        """Save transcription results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full transcription
        transcription_file = output_dir / f"{filename}_transcription.json"
        with open(transcription_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save plain text version
        text_file = output_dir / f"{filename}_transcription.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("ENHANCED TRANSCRIPTION:\n")
            f.write("=" * 50 + "\n")
            f.write(result["enhanced_text"])
            f.write("\n\n")
            f.write("ORIGINAL TRANSCRIPTION:\n")
            f.write("=" * 50 + "\n")
            f.write(result["original_text"])
        
        # Save test material summary
        if result["test_material"]:
            test_file = output_dir / f"{filename}_test_material.txt"
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("TEST MATERIAL DETECTED:\n")
                f.write("=" * 50 + "\n")
                for item in result["test_material"]:
                    f.write(f"[{item['start']:.1f}s - {item['end']:.1f}s] ")
                    f.write(f"({item['indicator_found']})\n")
                    f.write(f"{item['text']}\n\n")
        
        print(f"✓ Results saved to: {output_dir}")
        print(f"  - {transcription_file.name}")
        print(f"  - {text_file.name}")
        if result["test_material"]:
            print(f"  - {test_file.name}")
    
    def generate_summary(self, transcription_result: Dict, output_dir: Path) -> Optional[str]:
        """Generate AI summary of the lecture content."""
        if not self.client:
            print("WARNING: OpenAI client not available. Skipping AI summary.")
            return None
        
        try:
            print("Generating AI summary...")
            
            # Prepare context for summary
            context = f"""
            This is a military training lecture transcript from the Basic Officer Leader Course for Air Defense Artillery.
            
            Key context:
            - Focus on testable material and important concepts
            - Military acronyms and terminology are common
            - Look for definitions, procedures, and key principles
            - Emphasize anything marked as important or test material
            
            Military acronyms found: {', '.join(transcription_result['military_acronyms_found'])}
            
            Test material segments: {len(transcription_result['test_material'])}
            Key points identified: {len(transcription_result['key_points'])}
            
            TRANSCRIPT:
            {transcription_result['enhanced_text'][:4000]}  # Limit for API
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a military training assistant. Create a comprehensive summary of this lecture focusing on testable material, key concepts, and important definitions. Use bullet points and organize by topic."},
                    {"role": "user", "content": context}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            summary = response.choices[0].message.content
            
            # Save summary
            summary_file = output_dir / f"{transcription_result['audio_file'].replace('.m4a', '')}_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("LECTURE SUMMARY\n")
                f.write("=" * 50 + "\n")
                f.write(summary)
            
            print(f"✓ AI Summary saved to: {summary_file.name}")
            return summary
            
        except Exception as e:
            print(f"ERROR generating summary: {e}", file=sys.stderr)
            return None

def main():
    parser = argparse.ArgumentParser(description="Process military training audio with context awareness")
    parser.add_argument("audio_file", help="Path to audio file (.m4a, .mp3, .wav)")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"], 
                       help="Whisper model size (default: base)")
    parser.add_argument("--output-dir", default="downloads/audio/processed", 
                       help="Output directory for processed files")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--generate-summary", action="store_true", help="Generate AI summary")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("MILITARY AUDIO PROCESSOR")
    print("=" * 60)
    
    # Initialize processor
    processor = AudioProcessor(
        model_size=args.model,
        use_gpu=not args.no_gpu
    )
    
    # Process audio
    result = processor.transcribe_audio(audio_path, output_dir)
    
    # Generate summary if requested
    if args.generate_summary:
        processor.generate_summary(result, output_dir)
    
    # Print summary statistics
    print(f"\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Audio file: {audio_path.name}")
    print(f"  Duration: {result.get('duration', 'Unknown')}")
    print(f"  Word count: {result['processing_metadata']['word_count']}")
    print(f"  Military acronyms found: {len(result['military_acronyms_found'])}")
    print(f"  Test material segments: {len(result['test_material'])}")
    print(f"  Key points identified: {len(result['key_points'])}")

if __name__ == "__main__":
    main()
