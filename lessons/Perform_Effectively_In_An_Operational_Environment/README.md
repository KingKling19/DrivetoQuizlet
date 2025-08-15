# Perform Effectively In An Operational Environment

## Folder Structure
- `presentations/` - PowerPoint files (.pptx)
- `notes/` - Handwritten notes images (.png, .jpg, etc.)
- `audio/` - Lecture audio files (.m4a, .mp3, .wav)
- `processed/` - AI processing results
- `output/` - Final Quizlet flashcards (.tsv)

## Processing Commands
```bash
# Process PowerPoint only
python convert_folder_to_quizlet.py "downloads\lessons\Perform_Effectively_In_An_Operational_Environment/presentations"

# Process PowerPoint + Notes
python integrate_powerpoint_notes.py "downloads\lessons\Perform_Effectively_In_An_Operational_Environment/presentations/" "downloads\lessons\Perform_Effectively_In_An_Operational_Environment/notes/"

# Process PowerPoint + Audio
python integrate_powerpoint_audio.py "downloads\lessons\Perform_Effectively_In_An_Operational_Environment/presentations/" "downloads\lessons\Perform_Effectively_In_An_Operational_Environment/audio/"

# Process all three (PowerPoint + Notes + Audio)
python integrate_all_sources.py "downloads\lessons\Perform_Effectively_In_An_Operational_Environment"
```
