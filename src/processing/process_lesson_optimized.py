#!/usr/bin/env python3
"""
Optimized Lesson Processor with Background Model Preloading

This version includes:
- Background model preloading
- Progress indicators
- Lazy loading
- Better error handling
- Performance monitoring
"""

import argparse
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Environment variables loaded")
except ImportError:
    print("WARNING: python-dotenv not installed")
except Exception as e:
    print(f"WARNING: Could not load .env file: {e}")

class OptimizedLessonProcessor:
    def __init__(self, lesson_name: str, use_gpu: bool = True):
        """Initialize the optimized lesson processor."""
        self.lesson_name = lesson_name
        self.use_gpu = use_gpu
        self.start_time = time.time()
        
        # Initialize model manager and start background preloading
        self._initialize_models()
        
        # Setup lesson directory structure
        self.lesson_dir = Path(f"downloads/lessons/{lesson_name}")
        self._setup_directory_structure()
    
    def _initialize_models(self):
        """Initialize models in background for faster startup."""
        print("🚀 Initializing AI models in background...")
        
        try:
            from src.data.model_manager import model_manager
            
            # Start background preloading
            self.whisper_thread, self.ocr_thread = model_manager.preload_models(
                whisper_size="base",
                ocr_languages=['en'],
                use_gpu=self.use_gpu
            )
            
            print("✓ Background model preloading started")
            
        except ImportError:
            print("WARNING: Model manager not available, will load models on-demand")
            self.whisper_thread = None
            self.ocr_thread = None
    
    def _setup_directory_structure(self):
        """Setup the lesson directory structure."""
        print(f"📁 Setting up directory structure for: {self.lesson_name}")
        
        directories = [
            self.lesson_dir / "audio",
            self.lesson_dir / "notes", 
            self.lesson_dir / "presentations",
            self.lesson_dir / "processed",
            self.lesson_dir / "output"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("✓ Directory structure created")
    
    def _wait_for_models(self, timeout: int = 30):
        """Wait for background model loading to complete."""
        if self.whisper_thread and self.ocr_thread:
            print("⏳ Waiting for models to finish loading...")
            
            start_wait = time.time()
            while (self.whisper_thread.is_alive() or self.ocr_thread.is_alive()) and \
                  (time.time() - start_wait) < timeout:
                time.sleep(0.5)
            
            if self.whisper_thread.is_alive() or self.ocr_thread.is_alive():
                print("⚠️  Model loading timeout, continuing with on-demand loading")
            else:
                print("✓ Models loaded successfully")
    
    def _find_files(self) -> Dict[str, List[Path]]:
        """Find all relevant files for the lesson."""
        print("🔍 Scanning for lesson files...")
        
        files = {
            "audio": [],
            "notes": [],
            "presentations": [],
            "processed": []
        }
        
        # Look for audio files
        audio_patterns = ["*.mp3", "*.wav", "*.m4a", "*.flac"]
        for pattern in audio_patterns:
            files["audio"].extend(self.lesson_dir.glob(f"audio/{pattern}"))
        
        # Look for note images
        note_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        for pattern in note_patterns:
            files["notes"].extend(self.lesson_dir.glob(f"notes/{pattern}"))
        
        # Look for presentations
        presentation_patterns = ["*.pptx", "*.ppt"]
        for pattern in presentation_patterns:
            files["presentations"].extend(self.lesson_dir.glob(f"presentations/{pattern}"))
        
        # Look for processed files
        processed_patterns = ["*.json", "*.txt", "*.tsv"]
        for pattern in processed_patterns:
            files["processed"].extend(self.lesson_dir.glob(f"processed/{pattern}"))
        
        # Report findings
        total_files = sum(len(file_list) for file_list in files.values())
        print(f"✓ Found {total_files} files:")
        for category, file_list in files.items():
            if file_list:
                print(f"  - {category}: {len(file_list)} files")
        
        return files
    
    def process_notes(self, files: Dict[str, List[Path]]) -> bool:
        """Process handwritten notes with progress indicators."""
        if not files["notes"]:
            print("ℹ️  No note files found")
            return False
        
        print(f"📝 Processing {len(files['notes'])} note files...")
        
        try:
            from src.processing.notes_processor import NotesProcessor
            
            processor = NotesProcessor(use_gpu=self.use_gpu)
            notes_dir = self.lesson_dir / "notes"
            output_dir = self.lesson_dir / "processed"
            
            # Process each note with progress
            for i, note_file in enumerate(files["notes"], 1):
                print(f"  [{i}/{len(files['notes'])}] Processing: {note_file.name}")
                result = processor.extract_text_from_image(note_file)
                
                if result.get("error"):
                    print(f"    ⚠️  Error: {result['error']}")
                else:
                    print(f"    ✓ Extracted {result.get('word_count', 0)} words")
            
            # Process the entire folder
            summary = processor.process_notes_folder(notes_dir, output_dir)
            
            if summary:
                print(f"✓ Notes processing complete: {summary['total_words']} words extracted")
                return True
            else:
                print("⚠️  Notes processing failed")
                return False
                
        except ImportError:
            print("⚠️  Notes processing module not available")
            return False
        except Exception as e:
            print(f"✗ Notes processing error: {e}")
            return False
    
    def process_audio(self, files: Dict[str, List[Path]]) -> bool:
        """Process audio files with progress indicators."""
        if not files["audio"]:
            print("ℹ️  No audio files found")
            return False
        
        print(f"🎵 Processing {len(files['audio'])} audio files...")
        
        try:
            from src.processing.audio_processor import AudioProcessor
            
            processor = AudioProcessor(model_size="base", use_gpu=self.use_gpu)
            output_dir = self.lesson_dir / "processed"
            
            for i, audio_file in enumerate(files["audio"], 1):
                print(f"  [{i}/{len(files['audio'])}] Processing: {audio_file.name}")
                
                start_time = time.time()
                result = processor.transcribe_audio(audio_file, output_dir)
                processing_time = time.time() - start_time
                
                if result:
                    word_count = result['processing_metadata']['word_count']
                    print(f"    ✓ Transcribed {word_count} words in {processing_time:.1f}s")
                else:
                    print(f"    ✗ Transcription failed")
            
            return True
            
        except ImportError:
            print("⚠️  Audio processing module not available")
            return False
        except Exception as e:
            print(f"✗ Audio processing error: {e}")
            return False
    
    def integrate_sources(self, files: Dict[str, List[Path]]) -> Dict[str, Any]:
        """Integrate all sources with progress tracking."""
        print("🔗 Integrating all sources...")
        
        integration_data = {
            "lesson_name": self.lesson_name,
            "timestamp": datetime.now().isoformat(),
            "sources": {},
            "flashcards": []
        }
        
        # Load PowerPoint data
        quizlet_files = [f for f in files["processed"] if f.name.endswith(".quizlet.json")]
        if quizlet_files:
            try:
                with open(quizlet_files[0], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    powerpoint_flashcards = data
                else:
                    powerpoint_flashcards = data.get('cards', [])
                
                integration_data["sources"]["powerpoint"] = {
                    "flashcards": len(powerpoint_flashcards),
                    "file": quizlet_files[0].name
                }
                print(f"✓ Loaded {len(powerpoint_flashcards)} PowerPoint flashcards")
            except Exception as e:
                print(f"⚠️  Could not load PowerPoint data: {e}")
        
        # Load notes data
        notes_files = [f for f in files["processed"] if "notes_processed.json" in f.name]
        if notes_files:
            try:
                with open(notes_files[0], 'r', encoding='utf-8') as f:
                    notes_data = json.load(f)
                
                integration_data["sources"]["notes"] = {
                    "word_count": notes_data.get('total_words', 0),
                    "file": notes_files[0].name
                }
                print(f"✓ Loaded notes analysis: {notes_data.get('total_words', 0)} words")
            except Exception as e:
                print(f"⚠️  Could not load notes data: {e}")
        
        # Load audio data
        audio_files = [f for f in files["processed"] if "transcription.json" in f.name]
        if audio_files:
            try:
                with open(audio_files[0], 'r', encoding='utf-8') as f:
                    audio_data = json.load(f)
                
                integration_data["sources"]["audio"] = {
                    "word_count": audio_data.get('processing_metadata', {}).get('word_count', 0),
                    "file": audio_files[0].name
                }
                print(f"✓ Loaded audio transcription: {audio_data.get('processing_metadata', {}).get('word_count', 0)} words")
            except Exception as e:
                print(f"⚠️  Could not load audio data: {e}")
        
        # Create integrated output
        output_dir = self.lesson_dir / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Save integration summary
        summary_file = output_dir / f"{self.lesson_name}_integration_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(integration_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Integration summary saved: {summary_file}")
        return integration_data
    
    def process_lesson(self) -> bool:
        """Process the entire lesson with optimized performance."""
        print("=" * 60)
        print(f"🚀 PROCESSING LESSON: {self.lesson_name}")
        print("=" * 60)
        
        # Wait for background model loading
        self._wait_for_models()
        
        # Find all files
        files = self._find_files()
        
        # Process each component
        success = True
        
        # Process notes
        if files["notes"]:
            if not self.process_notes(files):
                success = False
        
        # Process audio
        if files["audio"]:
            if not self.process_audio(files):
                success = False
        
        # Integrate sources
        if any(files.values()):
            integration_result = self.integrate_sources(files)
        
        # Final summary
        total_time = time.time() - self.start_time
        print("=" * 60)
        print(f"✅ LESSON PROCESSING COMPLETE")
        print(f"⏱️  Total processing time: {total_time:.1f} seconds")
        print(f"📊 Success: {'Yes' if success else 'Partial'}")
        print("=" * 60)
        
        return success

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Optimized Lesson Processor")
    parser.add_argument("lesson_name", help="Name of the lesson to process")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--wait-timeout", type=int, default=30, 
                       help="Timeout for model loading (seconds)")
    
    args = parser.parse_args()
    
    try:
        processor = OptimizedLessonProcessor(args.lesson_name, use_gpu=not args.no_gpu)
        success = processor.process_lesson()
        
        if success:
            print("🎉 Lesson processing completed successfully!")
            sys.exit(0)
        else:
            print("⚠️  Lesson processing completed with some issues")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
