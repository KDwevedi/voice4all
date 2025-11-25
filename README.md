# HuggingFace Dataset Uploader - Voice for All

Python script for uploading the Gujarati TTS dataset to HuggingFace Hub in WebDataset format.

## Features

- Streams audio files directly from source without full download
- Creates TAR-based WebDataset shards (500 files per shard)
- Includes comprehensive metadata (speaker info, transcripts, domain, category)
- Memory-efficient processing with automatic cleanup

## Requirements

```bash
pip install huggingface-hub datasets
```

## Usage

```bash
export HF_TOKEN=your_huggingface_token
python upload_gujarati_tts.py Chakshu/gujarati-tts
```

For private datasets:
```bash
python upload_gujarati_tts.py Chakshu/gujarati-tts --private
```

## Dataset Structure

Each TAR shard contains paired files with matching prefixes:
- `.wav` files: Audio data (48kHz, 24-bit WAV format)
- `.json` files: Metadata including:
  - `text`: Gujarati transcription
  - `file_id`: Unique identifier
  - `category`: Category code (SPOR, AGRI, etc.)
  - `domain`: Full domain name
  - `speaker_id`: Speaker identifier
  - `speaker_gender`: Speaker gender
  - `speaker_age`: Speaker age
  - `language`: Language code (gu for Gujarati)

## Output

The script uploads to: `https://huggingface.co/datasets/{repo_id}`

Final dataset structure:
- Train split: ~8,242 files in 17 TAR shards
- Test split: ~858 files in 2 TAR shards
- Total: ~10.6 GB

## Source Data

SPICOR Gujarati TTS Corpus - 33.6 hours of domain-rich speech data across 19 domains.

## License

CC-BY-4.0
