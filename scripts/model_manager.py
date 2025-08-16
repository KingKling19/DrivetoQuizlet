#!/usr/bin/env python3
"""
Model Manager for Lazy Loading and Caching AI Models

This module provides efficient model management to reduce startup times
by implementing lazy loading and singleton patterns for heavy AI models.
"""

import threading
import time
import json
import pickle
from typing import Optional, Dict, Any, List
from pathlib import Path

class ModelManager:
    """Manages AI models with lazy loading and caching for improved performance."""
    
    # Class-level caches
    _whisper_models: Dict[str, Any] = {}
    _ocr_readers: Dict[str, Any] = {}
    _openai_clients: Dict[str, Any] = {}
    _cross_lesson_data: Optional[Dict[str, Any]] = None
    
    # Thread locks for thread-safe initialization
    _whisper_lock = threading.Lock()
    _ocr_lock = threading.Lock()
    _openai_lock = threading.Lock()
    _cross_lesson_lock = threading.Lock()
    
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
    
    @classmethod
    def get_cross_lesson_data(cls, config_dir: Path = Path("config")) -> Optional[Dict[str, Any]]:
        """Get or load cross-lesson analysis data for context enhancement."""
        if cls._cross_lesson_data is None:
            with cls._cross_lesson_lock:
                if cls._cross_lesson_data is None:
                    cls._cross_lesson_data = cls._load_cross_lesson_data(config_dir)
        
        return cls._cross_lesson_data
    
    @classmethod
    def _load_cross_lesson_data(cls, config_dir: Path) -> Optional[Dict[str, Any]]:
        """Load cross-lesson analysis data from files."""
        data = {
            "content_index": {},
            "semantic_embeddings": {},
            "lesson_relationships": {},
            "cross_references": {}
        }
        
        try:
            # Load content index
            index_file = config_dir / "lesson_content_index.json"
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    data["content_index"] = json.load(f)
            
            # Load semantic embeddings
            embeddings_file = config_dir / "semantic_embeddings.pkl"
            if embeddings_file.exists():
                with open(embeddings_file, 'rb') as f:
                    data["semantic_embeddings"] = pickle.load(f)
            
            # Load lesson relationships
            relationships_file = config_dir / "lesson_relationships_analysis.json"
            if relationships_file.exists():
                with open(relationships_file, 'r', encoding='utf-8') as f:
                    data["lesson_relationships"] = json.load(f)
            
            # Load cross-references
            cross_refs_file = config_dir / "cross_references.json"
            if cross_refs_file.exists():
                with open(cross_refs_file, 'r', encoding='utf-8') as f:
                    data["cross_references"] = json.load(f)
            
            print(f"✓ Loaded cross-lesson data: {len(data['content_index'])} lessons indexed")
            return data
        except Exception as e:
            print(f"⚠️  Could not load cross-lesson data: {e}")
            return None
    
    @classmethod
    def get_related_lessons(cls, lesson_id: str, max_lessons: int = 3) -> List[Dict[str, Any]]:
        """Get lessons related to the specified lesson for context enhancement."""
        cross_lesson_data = cls.get_cross_lesson_data()
        if not cross_lesson_data:
            return []
        
        try:
            relationships = cross_lesson_data.get("lesson_relationships", {})
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
            result = []
            for rel in related[:max_lessons]:
                if isinstance(rel, dict):
                    related_lesson_id = rel.get("lesson_id")
                    if related_lesson_id and related_lesson_id != lesson_id:
                        result.append(rel)
            
            return result
        except Exception as e:
            print(f"⚠️  Error getting related lessons: {e}")
            return []
    
    @classmethod
    def get_lesson_context(cls, lesson_id: str, max_context_length: int = 2000) -> str:
        """Get context information for a specific lesson."""
        cross_lesson_data = cls.get_cross_lesson_data()
        if not cross_lesson_data:
            return ""
        
        try:
            content_index = cross_lesson_data.get("content_index", {})
            lesson_data = content_index.get(lesson_id, {})
            
            if not lesson_data or not isinstance(lesson_data, dict):
                return ""
            
            context_parts = []
            lesson_name = lesson_data.get("lesson_name", lesson_id)
            
            # Add lesson overview
            context_parts.append(f"## Lesson: {lesson_name}")
            
            # Add key concepts
            key_concepts = lesson_data.get("key_concepts", [])
            if key_concepts and isinstance(key_concepts, list):
                context_parts.append("### Key Concepts:")
                for concept in key_concepts[:10]:  # Limit to top 10 concepts
                    context_parts.append(f"- {concept}")
            
            # Add prerequisites
            prerequisites = lesson_data.get("prerequisites", [])
            if prerequisites and isinstance(prerequisites, list):
                context_parts.append("### Prerequisites:")
                for prereq in prerequisites[:5]:  # Limit to top 5 prerequisites
                    context_parts.append(f"- {prereq}")
            
            # Add content snippets
            content_sources = lesson_data.get("content_sources", {})
            if isinstance(content_sources, dict):
                presentations = content_sources.get("presentations", {})
                if isinstance(presentations, dict):
                    for pptx_name, pptx_data in list(presentations.items())[:2]:  # Limit to 2 presentations
                        if isinstance(pptx_data, dict):
                            slides = pptx_data.get("slides", [])
                            if isinstance(slides, list):
                                for slide in slides[:3]:  # Limit to 3 slides per presentation
                                    if isinstance(slide, dict):
                                        title = slide.get("title", "")
                                        body = slide.get("body", "")
                                        if title and body:
                                            context_parts.append(f"### {title}")
                                            context_parts.append(body[:200] + "..." if len(body) > 200 else body)
            
            context = "\n".join(context_parts)
            
            # Limit total context length
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
            
            return context
        except Exception as e:
            print(f"⚠️  Error getting lesson context: {e}")
            return ""
    
    @classmethod
    def calculate_similarity_score(cls, lesson_id_1: str, lesson_id_2: str) -> float:
        """Calculate similarity score between two lessons."""
        cross_lesson_data = cls.get_cross_lesson_data()
        if not cross_lesson_data:
            return 0.0
        
        try:
            relationships = cross_lesson_data.get("lesson_relationships", {})
            lesson_rels = relationships.get(lesson_id_1, {})
            related_lessons = lesson_rels.get("related_lessons", [])
            
            for rel in related_lessons:
                if rel.get("lesson_id") == lesson_id_2:
                    return rel.get("similarity_score", 0.0)
            
            return 0.0
        except Exception as e:
            print(f"⚠️  Error calculating similarity score: {e}")
            return 0.0
    
    @classmethod
    def clear_cross_lesson_cache(cls):
        """Clear cross-lesson data cache."""
        with cls._cross_lesson_lock:
            cls._cross_lesson_data = None
        print("✓ Cross-lesson data cache cleared")

# Global instance for easy access
model_manager = ModelManager()
