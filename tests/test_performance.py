#!/usr/bin/env python3
"""
Performance Test Script

Demonstrates the performance improvements from the optimized system
by comparing old vs new loading times.
"""

import time
import sys
from pathlib import Path

def test_old_loading():
    """Test the old loading method (immediate model loading)."""
    print("🔄 Testing OLD loading method (immediate model loading)...")
    start_time = time.time()
    
    try:
        # Simulate old audio processor initialization
        print("  Initializing AudioProcessor...")
        from audio_processor import AudioProcessor
        processor = AudioProcessor(model_size="base", use_gpu=True)
        
        # Simulate old notes processor initialization
        print("  Initializing NotesProcessor...")
        from notes_processor import NotesProcessor
        notes_processor = NotesProcessor(use_gpu=True)
        
        total_time = time.time() - start_time
        print(f"✓ OLD method completed in {total_time:.2f}s")
        return total_time
        
    except Exception as e:
        print(f"✗ OLD method failed: {e}")
        return None

def test_new_loading():
    """Test the new loading method (lazy loading with model manager)."""
    print("🚀 Testing NEW loading method (lazy loading)...")
    start_time = time.time()
    
    try:
        # Initialize with lazy loading
        print("  Initializing AudioProcessor (lazy)...")
        from audio_processor import AudioProcessor
        processor = AudioProcessor(model_size="base", use_gpu=True)
        
        print("  Initializing NotesProcessor (lazy)...")
        from notes_processor import NotesProcessor
        notes_processor = NotesProcessor(use_gpu=True)
        
        init_time = time.time() - start_time
        print(f"✓ NEW method initialization completed in {init_time:.2f}s")
        
        # Now trigger actual model loading
        print("  Triggering model loading...")
        model_start = time.time()
        
        # Load Whisper model
        processor.load_model()
        whisper_time = time.time() - model_start
        
        # Load OCR model
        ocr_start = time.time()
        notes_processor.extract_text_from_image(Path("test.jpg"))  # This will trigger OCR loading
        ocr_time = time.time() - ocr_start
        
        total_time = time.time() - start_time
        print(f"✓ NEW method completed in {total_time:.2f}s")
        print(f"  - Initialization: {init_time:.2f}s")
        print(f"  - Whisper loading: {whisper_time:.2f}s")
        print(f"  - OCR loading: {ocr_time:.2f}s")
        
        return total_time, init_time
        
    except Exception as e:
        print(f"✗ NEW method failed: {e}")
        return None, None

def test_model_manager():
    """Test the model manager caching."""
    print("🧠 Testing Model Manager caching...")
    
    try:
        from model_manager import model_manager
        
        # First load
        print("  First Whisper model load...")
        start_time = time.time()
        model1 = model_manager.get_whisper_model("base", True)
        first_load = time.time() - start_time
        print(f"    ✓ First load: {first_load:.2f}s")
        
        # Second load (should be cached)
        print("  Second Whisper model load (cached)...")
        start_time = time.time()
        model2 = model_manager.get_whisper_model("base", True)
        second_load = time.time() - start_time
        print(f"    ✓ Second load: {second_load:.2f}s")
        
        # Check if it's the same model instance
        is_same = model1 is model2
        print(f"    ✓ Same instance: {is_same}")
        
        # Show cache status
        status = model_manager.get_cache_status()
        print(f"    ✓ Cache status: {status}")
        
        return first_load, second_load
        
    except Exception as e:
        print(f"✗ Model manager test failed: {e}")
        return None, None

def main():
    """Run performance comparison tests."""
    print("=" * 60)
    print("🏁 PERFORMANCE COMPARISON TEST")
    print("=" * 60)
    
    # Test old vs new loading
    print("\n1️⃣ Testing Loading Methods:")
    old_time = test_old_loading()
    print()
    new_time, init_time = test_new_loading()
    
    if old_time and new_time:
        improvement = ((old_time - new_time) / old_time) * 100
        print(f"\n📊 Loading Time Improvement: {improvement:.1f}% faster")
        print(f"   Old: {old_time:.2f}s")
        print(f"   New: {new_time:.2f}s")
        print(f"   Initialization: {init_time:.2f}s")
    
    # Test model manager caching
    print("\n2️⃣ Testing Model Manager Caching:")
    first_load, second_load = test_model_manager()
    
    if first_load and second_load:
        cache_speedup = first_load / second_load if second_load > 0 else float('inf')
        print(f"\n📊 Cache Speedup: {cache_speedup:.1f}x faster for cached models")
    
    # Test performance monitoring
    print("\n3️⃣ Testing Performance Monitoring:")
    try:
        from performance_monitor import performance_monitor
        
        performance_monitor.start_monitoring()
        time.sleep(2)  # Monitor for 2 seconds
        performance_monitor.stop_monitoring()
        
        performance_monitor.print_summary()
        
    except Exception as e:
        print(f"✗ Performance monitoring test failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Performance test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
