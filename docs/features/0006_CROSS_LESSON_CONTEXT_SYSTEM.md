# Feature 6: Cross-Lesson Context System - Technical Plan

## Feature Description
Implement a system to use neighboring lesson content for better context during AI analysis and flashcard generation. This will enhance the quality of content analysis by providing cross-lesson references, related concepts, and contextual information from other lessons in the curriculum.

## Current State Analysis

### Existing Components
- `scripts/organize_lessons.py` - Basic lesson organization and structure creation
- `scripts/batch_process_lessons.py` - Batch processing across multiple lessons
- `scripts/convert_folder_to_quizlet.py` - Main flashcard generation with basic slide windowing
- `scripts/model_manager.py` - AI model management and caching
- `lessons/` directory structure with organized content (TLP, Conducting_Operations_in_a_Degraded_Space, Perform_Effectively_In_An_Operational_Environment)

### Current Limitations
- No cross-lesson content analysis or similarity detection
- Limited context window (only neighboring slides within same presentation)
- No lesson relationship mapping or content correlation
- No intelligent cross-reference detection between lessons
- No context weighting system for prioritizing related content

## Technical Requirements

### Phase 1: Data Layer & Content Indexing

#### Files to Create/Modify:
1. **`scripts/lesson_content_indexer.py`** - New file
   - Extract and index content from all lessons
   - Create searchable content database
   - Generate content fingerprints for similarity detection
   - Store metadata about lesson relationships

2. **`scripts/cross_lesson_analyzer.py`** - New file
   - Analyze content similarities between lessons
   - Detect related concepts and cross-references
   - Generate lesson relationship graph
   - Create context enhancement recommendations

3. **`config/lesson_relationships.json`** - New file
   - Store lesson hierarchy and relationships
   - Define content correlation weights
   - Configure context window parameters

#### Algorithms:
1. **Content Fingerprinting Algorithm**
   - Extract key terms, concepts, and definitions from each lesson
   - Generate TF-IDF vectors for content comparison
   - Create semantic embeddings using OpenAI embeddings API
   - Store fingerprints in searchable database

2. **Similarity Detection Algorithm**
   - Calculate cosine similarity between lesson content vectors
   - Identify overlapping concepts and terminology
   - Detect prerequisite relationships between lessons
   - Score content relevance using weighted factors

### Phase 2A: Context Enhancement Engine

#### Files to Modify:
1. **`scripts/convert_folder_to_quizlet.py`**
   - Enhance `extract_slide_text()` function to include cross-lesson context
   - Modify `process_slides()` to incorporate related content
   - Update `SYSTEM_PROMPT` to include cross-lesson awareness
   - Add context window expansion for related lesson content

2. **`scripts/model_manager.py`**
   - Add cross-lesson context retrieval methods
   - Implement context caching for performance
   - Add similarity scoring utilities

#### Algorithms:
1. **Context Window Expansion Algorithm**
   - Identify related content from other lessons
   - Weight context by relevance and relationship strength
   - Expand context window beyond current lesson boundaries
   - Prioritize context based on concept importance

2. **Cross-Reference Detection Algorithm**
   - Scan for concept mentions across lessons
   - Identify definition variations and clarifications
   - Detect prerequisite knowledge requirements
   - Map concept evolution across curriculum

### Phase 2B: Enhanced Processing Pipeline

#### Files to Modify:
1. **`scripts/integrate_powerpoint_notes.py`**
   - Add cross-lesson context to notes integration
   - Enhance content correlation with related lessons

2. **`scripts/integrate_powerpoint_audio.py`**
   - Include cross-lesson context in audio integration
   - Improve transcription accuracy with related content

3. **`scripts/process_lesson.py`**
   - Integrate cross-lesson analysis into main processing pipeline
   - Add context enhancement as processing step

#### Algorithms:
1. **Context Weighting Algorithm**
   - Score context relevance based on multiple factors:
     - Semantic similarity to current content
     - Lesson relationship strength
     - Concept importance and frequency
     - Temporal relationship (prerequisites)
   - Apply weighted context to enhance AI analysis

2. **Content Correlation Algorithm**
   - Map related concepts across lessons
   - Identify complementary definitions and explanations
   - Detect concept variations and clarifications
   - Create unified concept understanding

### Phase 3: Advanced Context Features

#### Files to Create:
1. **`scripts/context_optimizer.py`** - New file
   - Optimize context selection for maximum relevance
   - Implement adaptive context window sizing
   - Provide context quality scoring

2. **`scripts/lesson_relationship_visualizer.py`** - New file
   - Generate visual maps of lesson relationships
   - Show content correlation networks
   - Provide insights into curriculum structure

#### Algorithms:
1. **Adaptive Context Selection Algorithm**
   - Dynamically adjust context window size based on content complexity
   - Optimize context relevance for specific content types
   - Balance context depth vs. processing efficiency

2. **Context Quality Assessment Algorithm**
   - Score context relevance and usefulness
   - Identify optimal context combinations
   - Provide feedback for context improvement

## Implementation Details

### Data Structures
```python
# Lesson content index
{
    "lesson_id": str,
    "content_fingerprint": List[float],  # TF-IDF vector
    "semantic_embedding": List[float],   # OpenAI embedding
    "key_concepts": List[str],
    "prerequisites": List[str],
    "related_lessons": List[str],
    "content_metadata": Dict
}

# Context enhancement data
{
    "source_lesson": str,
    "target_lesson": str,
    "similarity_score": float,
    "related_concepts": List[str],
    "context_weight": float,
    "relationship_type": str  # "prerequisite", "related", "complementary"
}
```

### API Integration Points
- OpenAI Embeddings API for semantic similarity
- Existing OpenAI GPT-4o-mini for enhanced context-aware analysis
- Local TF-IDF processing for content fingerprinting

### Performance Considerations
- Implement caching for similarity calculations
- Use batch processing for large lesson sets
- Optimize context retrieval with indexing
- Implement progressive context loading

### Error Handling
- Graceful fallback when cross-lesson context unavailable
- Validation of context relevance and quality
- Error recovery for failed similarity calculations
- Logging and monitoring of context enhancement effectiveness

## Dependencies
- Feature 4 (Content Extraction & Normalization) - Required
- Feature 5 (AI Content Analysis Engine) - Required
- OpenAI API access for embeddings and enhanced analysis
- Existing lesson organization structure

## Success Criteria
- Cross-lesson context successfully integrated into flashcard generation
- Improved content analysis quality through enhanced context
- Reduced duplicate or conflicting definitions across lessons
- Better concept relationships and prerequisite mapping
- Maintained processing performance with context enhancement
