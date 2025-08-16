# Feature 9: Quizlet API Integration

## Overview
Enable automatic upload of generated flashcards to Quizlet to complete the end-to-end workflow from Google Drive content to Quizlet study sets. This feature will integrate with the existing flashcard generation system and provide both programmatic and web dashboard interfaces for managing Quizlet uploads.

## Current State Analysis
- Flashcard generation system (`scripts/convert_folder_to_quizlet.py`) produces TSV files ready for Quizlet import
- Output format: `term\tdefinition` tab-separated values
- Web dashboard (`scripts/enhanced_dashboard.py`) has comprehensive API endpoints for lesson management
- Configuration system (`config/drive_config.json`) exists for API credentials and settings
- No existing Quizlet API integration or authentication

## Files to Create/Modify

### New Files:
1. **`scripts/quizlet_api.py`** - Main Quizlet API integration module
2. **`scripts/quizlet_uploader.py`** - Flashcard upload orchestration
3. **`config/quizlet_config.json`** - Quizlet API configuration
4. **`scripts/test_quizlet_integration.py`** - Integration testing

### Files to Modify:
1. **`scripts/enhanced_dashboard.py`** - Add Quizlet API endpoints
2. **`scripts/convert_folder_to_quizlet.py`** - Add optional direct upload capability
3. **`templates/enhanced_dashboard.html`** - Add Quizlet management UI
4. **`static/js/dashboard.js`** - Add Quizlet upload functionality
5. **`requirements.txt`** - Add Quizlet API dependencies

## Technical Implementation

### Phase 1: Data Layer & Configuration

#### 1.1 Quizlet Configuration System
**File**: `config/quizlet_config.json`
```json
{
  "api": {
    "client_id": "",
    "client_secret": "",
    "redirect_uri": "http://localhost:8000/auth/quizlet/callback",
    "scope": "write_set"
  },
  "upload": {
    "default_visibility": "public",
    "auto_create_sets": true,
    "set_naming": "lesson_name_flashcards",
    "max_cards_per_set": 1000,
    "retry_attempts": 3,
    "retry_delay": 5
  },
  "authentication": {
    "token_file": "config/quizlet_token.json",
    "refresh_threshold_hours": 1
  }
}
```

#### 1.2 Database Schema Extensions
**File**: `config/quizlet_uploads.db`
```sql
CREATE TABLE quizlet_sets (
    id INTEGER PRIMARY KEY,
    lesson_name TEXT NOT NULL,
    quizlet_set_id TEXT UNIQUE,
    quizlet_set_url TEXT,
    card_count INTEGER,
    upload_status TEXT DEFAULT 'pending',
    upload_timestamp DATETIME,
    last_sync DATETIME,
    error_message TEXT
);

CREATE TABLE quizlet_upload_queue (
    id INTEGER PRIMARY KEY,
    lesson_name TEXT NOT NULL,
    tsv_file_path TEXT NOT NULL,
    priority INTEGER DEFAULT 0,
    status TEXT DEFAULT 'queued',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    processed_at DATETIME,
    error_message TEXT
);
```

### Phase 2A: Core API Integration

#### 2.1 Quizlet API Client
**File**: `scripts/quizlet_api.py`

**Key Functions**:
- `QuizletAPI` class with OAuth2 authentication
- `authenticate()` - Handle OAuth2 flow with refresh token support
- `create_set(title, description, cards)` - Create new Quizlet set
- `update_set(set_id, cards)` - Update existing set
- `get_set(set_id)` - Retrieve set information
- `delete_set(set_id)` - Remove set
- `search_sets(query)` - Find existing sets by name

**Authentication Flow**:
1. User initiates authentication via dashboard
2. Redirect to Quizlet OAuth2 authorization URL
3. Handle callback and store access/refresh tokens
4. Implement token refresh logic for expired tokens

**API Integration Points**:
- Quizlet API v2.0 endpoints for set management
- OAuth2 authentication with PKCE flow
- Rate limiting and error handling
- Retry logic with exponential backoff

#### 2.2 Flashcard Upload Orchestrator
**File**: `scripts/quizlet_uploader.py`

**Key Functions**:
- `QuizletUploader` class for managing upload workflow
- `upload_lesson_flashcards(lesson_name, tsv_file_path)` - Main upload function
- `process_upload_queue()` - Background queue processor
- `validate_flashcards(tsv_file_path)` - Pre-upload validation
- `split_large_sets(cards, max_cards)` - Handle Quizlet's card limits
- `generate_set_metadata(lesson_name)` - Create set titles and descriptions

**Upload Workflow**:
1. Validate TSV file format and content
2. Check for existing Quizlet set for lesson
3. Split cards if exceeding Quizlet limits (1000 cards per set)
4. Create or update Quizlet set(s)
5. Update database with upload status
6. Return Quizlet set URLs for user access

### Phase 2B: Web Dashboard Integration

#### 2.3 Dashboard API Endpoints
**File**: `scripts/enhanced_dashboard.py`

**New Endpoints**:
```python
@app.get("/api/quizlet/auth/url")
@app.get("/api/quizlet/auth/callback")
@app.post("/api/quizlet/auth/refresh")
@app.get("/api/quizlet/sets")
@app.post("/api/quizlet/upload/{lesson_name}")
@app.get("/api/quizlet/upload/{lesson_name}/status")
@app.delete("/api/quizlet/sets/{set_id}")
@app.get("/api/quizlet/queue")
@app.post("/api/quizlet/queue/process")
```

**Authentication Endpoints**:
- `/api/quizlet/auth/url` - Generate OAuth2 authorization URL
- `/api/quizlet/auth/callback` - Handle OAuth2 callback
- `/api/quizlet/auth/refresh` - Refresh expired tokens

**Upload Management Endpoints**:
- `/api/quizlet/upload/{lesson_name}` - Upload lesson flashcards
- `/api/quizlet/upload/{lesson_name}/status` - Check upload progress
- `/api/quizlet/sets` - List all uploaded sets
- `/api/quizlet/queue` - View upload queue status

#### 2.4 Dashboard UI Components
**File**: `templates/enhanced_dashboard.html`

**New UI Sections**:
- Quizlet Authentication Status Panel
- Upload Queue Management Interface
- Set Management Dashboard
- Upload Progress Indicators
- Error Display and Retry Controls

**JavaScript Functions** (`static/js/dashboard.js`):
- `initQuizletAuth()` - Initialize authentication flow
- `uploadToQuizlet(lessonName)` - Trigger upload
- `monitorUploadProgress(lessonName)` - Track upload status
- `refreshQuizletSets()` - Update set list
- `retryFailedUpload(uploadId)` - Retry failed uploads

### Phase 2C: Enhanced Flashcard Generation

#### 2.5 Direct Upload Integration
**File**: `scripts/convert_folder_to_quizlet.py`

**Modifications**:
- Add `--upload-to-quizlet` command line flag
- Add `--quizlet-set-name` parameter for custom set naming
- Integrate with `QuizletUploader` for direct upload after generation
- Add upload status reporting to existing verbose output

**New Function**:
```python
def upload_generated_flashcards(cards: List[Dict], lesson_name: str, set_name: Optional[str] = None) -> Dict[str, Any]:
    """Upload generated flashcards directly to Quizlet"""
```

## Algorithms & Workflows

### 1. OAuth2 Authentication Flow
1. User clicks "Connect Quizlet" in dashboard
2. Generate authorization URL with PKCE challenge
3. Redirect user to Quizlet authorization page
4. Handle callback with authorization code
5. Exchange code for access/refresh tokens
6. Store tokens securely in `config/quizlet_token.json`
7. Implement automatic token refresh before API calls

### 2. Flashcard Upload Workflow
1. **Validation**: Check TSV file format and content quality
2. **Set Management**: Check for existing Quizlet set with same name
3. **Card Processing**: 
   - Clean and validate each card (term/definition)
   - Remove duplicates and low-quality cards
   - Split into chunks if exceeding 1000 cards per set
4. **Upload**: Create or update Quizlet set(s)
5. **Status Tracking**: Update database with upload results
6. **Notification**: Provide user with Quizlet set URLs

### 3. Queue Processing System
1. **Queue Management**: Background processor for pending uploads
2. **Priority Handling**: Process high-priority uploads first
3. **Error Recovery**: Automatic retry with exponential backoff
4. **Status Updates**: Real-time progress reporting via WebSocket
5. **Cleanup**: Remove completed/failed items from queue

### 4. Set Naming Convention
- **Format**: `{Lesson_Name}_Flashcards_{Date}`
- **Examples**: 
  - `TLP_Flashcards_2024-01-15`
  - `Conducting_Operations_Degraded_Space_Flashcards_2024-01-15`
- **Handling**: Replace spaces with underscores, limit length to 100 characters

## Error Handling & Edge Cases

### 1. Authentication Errors
- Token expiration handling with automatic refresh
- Invalid credentials with user re-authentication flow
- Network connectivity issues with retry logic

### 2. Upload Errors
- Rate limiting with exponential backoff
- Large file handling with chunking
- Duplicate set detection and resolution
- Invalid card format filtering

### 3. Data Validation
- TSV format validation before upload
- Card content quality assessment
- Character limit enforcement for Quizlet API
- Special character handling and encoding

## Dependencies & Requirements

### New Dependencies (`requirements.txt`):
```
requests-oauthlib>=1.3.0
aiohttp>=3.8.0
websockets>=10.0
```

### External Dependencies:
- Quizlet API v2.0 access
- OAuth2 application registration on Quizlet
- Valid Quizlet developer account

## Testing Strategy

### 1. Unit Tests (`scripts/test_quizlet_integration.py`)
- OAuth2 authentication flow testing
- API endpoint testing with mock responses
- Upload workflow testing with sample data
- Error handling and retry logic testing

### 2. Integration Tests
- End-to-end upload workflow testing
- Dashboard integration testing
- Queue processing system testing
- Cross-browser compatibility testing

### 3. Performance Testing
- Large file upload performance
- Concurrent upload handling
- Rate limiting compliance testing
- Memory usage optimization

## Security Considerations

### 1. Token Security
- Secure storage of OAuth2 tokens
- Token encryption at rest
- Automatic token rotation
- Secure token transmission

### 2. API Security
- Rate limiting compliance
- Input validation and sanitization
- Error message sanitization
- Secure callback URL handling

### 3. Data Privacy
- Minimal data retention policies
- User consent for Quizlet integration
- Secure deletion of sensitive data
- Audit logging for upload activities

## Success Criteria
- Successfully authenticate with Quizlet API
- Upload flashcards from any lesson to Quizlet
- Provide real-time upload status and progress
- Handle large flashcard sets (>1000 cards)
- Integrate seamlessly with existing dashboard
- Support automatic retry for failed uploads
- Maintain upload history and set management

