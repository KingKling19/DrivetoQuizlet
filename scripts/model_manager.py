#!/usr/bin/env python3
"""
Model Manager for Lazy Loading and Caching AI Models

This module provides efficient model management to reduce startup times
by implementing lazy loading and singleton patterns for heavy AI models.
"""

import threading
import time
from typing import Optional, Dict, Any
from pathlib import Path

class ModelManager:
    """Manages AI models with lazy loading and caching for improved performance."""
    
    # Class-level caches
    _whisper_models: Dict[str, Any] = {}
    _ocr_readers: Dict[str, Any] = {}
    _openai_clients: Dict[str, Any] = {}
    
    # Thread locks for thread-safe initialization
    _whisper_lock = threading.Lock()
    _ocr_lock = threading.Lock()
    _openai_lock = threading.Lock()
    
    @classmethod
    def get_whisper_model(cls, model_size: str = "base", use_gpu: bool = True) -> Any:
        """Get or load Whisper model with lazy loading."""
        cache_key = f"{model_size}_{'gpu' if use_gpu else 'cpu'}"
        
        if cache_key not in cls._whisper_models:
            with cls._whisper_lock:
                # Double-check pattern to avoid race conditions
                if cache_key not in cls._whisper_models:
                    print(f"🔄 Loading Whisper model: {model_size}")
                    start_time = time.time()
                    
                    try:
                        import whisper
                        model = whisper.load_model(model_size)
                        device = "GPU" if model.device.type == "cuda" else "CPU"
                        load_time = time.time() - start_time
                        print(f"✓ Whisper {model_size} loaded on {device} in {load_time:.2f}s")
                        
                        cls._whisper_models[cache_key] = model
                    except Exception as e:
                        print(f"✗ Failed to load Whisper model: {e}")
                        raise
        
        return cls._whisper_models[cache_key]
    
    @classmethod
    def get_ocr_reader(cls, languages: list = None, use_gpu: bool = True) -> Any:
        """Get or load EasyOCR reader with lazy loading."""
        if languages is None:
            languages = ['en']
        
        cache_key = f"{'_'.join(languages)}_{'gpu' if use_gpu else 'cpu'}"
        
        if cache_key not in cls._ocr_readers:
            with cls._ocr_lock:
                if cache_key not in cls._ocr_readers:
                    print(f"🔄 Initializing OCR reader for languages: {languages}")
                    start_time = time.time()
                    
                    try:
                        import easyocr
                        reader = easyocr.Reader(languages, gpu=use_gpu)
                        load_time = time.time() - start_time
                        print(f"✓ OCR reader initialized in {load_time:.2f}s")
                        
                        cls._ocr_readers[cache_key] = reader
                    except Exception as e:
                        print(f"✗ Failed to initialize OCR reader: {e}")
                        raise
        
        return cls._ocr_readers[cache_key]
    
    @classmethod
    def get_openai_client(cls, api_key: str = None) -> Any:
        """Get or create OpenAI client."""
        if api_key is None:
            import os
            api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        cache_key = api_key[:10] + "..."  # Use partial key for caching
        
        if cache_key not in cls._openai_clients:
            with cls._openai_lock:
                if cache_key not in cls._openai_clients:
                    print("🔄 Initializing OpenAI client...")
                    start_time = time.time()
                    
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=api_key)
                        load_time = time.time() - start_time
                        print(f"✓ OpenAI client initialized in {load_time:.2f}s")
                        
                        cls._openai_clients[cache_key] = client
                    except Exception as e:
                        print(f"✗ Failed to initialize OpenAI client: {e}")
                        raise
        
        return cls._openai_clients[cache_key]
    
    @classmethod
    def preload_models(cls, whisper_size: str = "base", ocr_languages: list = None, use_gpu: bool = True):
        """Preload commonly used models in background threads."""
        print("🚀 Preloading AI models in background...")
        
        def preload_whisper():
            try:
                cls.get_whisper_model(whisper_size, use_gpu)
            except Exception as e:
                print(f"WARNING: Whisper preload failed: {e}")
        
        def preload_ocr():
            try:
                if ocr_languages is None:
                    ocr_languages = ['en']
                cls.get_ocr_reader(ocr_languages, use_gpu)
            except Exception as e:
                print(f"WARNING: OCR preload failed: {e}")
        
        # Start background threads for preloading
        whisper_thread = threading.Thread(target=preload_whisper, daemon=True)
        ocr_thread = threading.Thread(target=preload_ocr, daemon=True)
        
        whisper_thread.start()
        ocr_thread.start()
        
        return whisper_thread, ocr_thread
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached models (useful for memory management)."""
        print("🧹 Clearing model cache...")
        cls._whisper_models.clear()
        cls._ocr_readers.clear()
        cls._openai_clients.clear()
        
        # Clear GPU memory if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✓ GPU memory cleared")
        except ImportError:
            pass
    
    @classmethod
    def get_cache_status(cls) -> Dict[str, Any]:
        """Get status of cached models."""
        return {
            "whisper_models": list(cls._whisper_models.keys()),
            "ocr_readers": list(cls._ocr_readers.keys()),
            "openai_clients": len(cls._openai_clients),
            "total_cached": len(cls._whisper_models) + len(cls._ocr_readers) + len(cls._openai_clients)
        }

# Global instance for easy access
model_manager = ModelManager()
