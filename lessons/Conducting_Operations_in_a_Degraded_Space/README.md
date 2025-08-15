# Conducting Operations in a Degraded Space

## Folder Structure
- `presentations/` - PowerPoint files (.pptx)
- `notes/` - Handwritten notes images (.png, .jpg, etc.)
- `audio/` - Lecture audio files (.m4a, .mp3, .wav)
- `processed/` - AI processing results
- `output/` - Final Quizlet flashcards (.tsv)

## Processing Commands
```bash
# Process PowerPoint only
python convert_folder_to_quizlet.py "downloads\lessons\Conducting_Operations_in_a_Degraded_Space/presentations"

# Process PowerPoint + Notes
python integrate_powerpoint_notes.py "downloads\lessons\Conducting_Operations_in_a_Degraded_Space/presentations/" "downloads\lessons\Conducting_Operations_in_a_Degraded_Space/notes/"

# Process PowerPoint + Audio
python integrate_powerpoint_audio.py "downloads\lessons\Conducting_Operations_in_a_Degraded_Space/presentations/" "downloads\lessons\Conducting_Operations_in_a_Degraded_Space/audio/"

# Process all three (PowerPoint + Notes + Audio)
python integrate_all_sources.py "downloads\lessons\Conducting_Operations_in_a_Degraded_Space"
```
