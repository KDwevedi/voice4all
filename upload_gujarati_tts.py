#!/usr/bin/env python3
"""
Gujarati TTS dataset - WebDataset format (TAR-based) - FIXED VERSION
Includes speaker metadata and correct file extensions
"""

import json
import os
import tempfile
import tarfile
import subprocess
from pathlib import Path
from typing import Generator
from huggingface_hub import HfApi, create_repo

URLS = {
    "train": "https://objectstore.e2enetworks.net/iisc-spire-corpora/spicor/gujarati_tts/IISc_SPICORProject_Gujarati_Female_Spk001_HC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20251124%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251124T060534Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=e12850f90ea0e97da9b6ca4e9f38ca886a3ce313274f24b8489caa8ffbaad0ec",
    "test": "https://objectstore.e2enetworks.net/iisc-spire-corpora/spicor/gujarati_tts/IISc_SPICORProject_Gujarati_Female_Spk001_NHC.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=0U8R7S2207455OVWTNCN%2F20251124%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251124T060534Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=613aed26828e6b334aaa894e11e97b30f74c18cfd4a597094007fbce45f05392"
}

SHARD_SIZE = 500  # Files per TAR shard

# Speaker metadata from README
SPEAKER_METADATA = {
    "speaker_id": "Spk0001",
    "speaker_gender": "Female",
    "speaker_age": 33,
    "language": "gu",  # Gujarati
}


def stream_tar_members(url: str) -> Generator:
    """Stream tar members one by one"""
    print(f"Streaming from source...")
    proc = subprocess.Popen(
        f'wget -q -O - "{url}"',
        shell=True,
        stdout=subprocess.PIPE
    )

    with tarfile.open(fileobj=proc.stdout, mode='r|gz') as tar:
        for member in tar:
            if member.isfile():
                yield member, tar.extractfile(member)

    proc.wait()


def process_split(url: str, split_name: str, repo_id: str, api: HfApi):
    """Process split by creating WebDataset TAR shards"""
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split")
    print(f"{'='*60}\n")

    transcripts = {}
    shard_num = 0
    shard_files = []
    total_files = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for member, fileobj in stream_tar_members(url):
            filename = Path(member.name).name

            # Load transcripts
            if filename.endswith('_Transcripts.json'):
                print(f"Loading transcripts...")
                data = json.loads(fileobj.read())
                transcripts = data.get('Transcripts', {})
                print(f"✓ Loaded {len(transcripts)} transcripts\n")
                continue

            # Process WAV files
            if not filename.endswith('.wav'):
                continue

            # Get metadata
            file_id = Path(filename).stem
            transcript = transcripts.get(file_id, {})
            text = transcript.get("Transcript", "") if isinstance(transcript, dict) else ""
            domain = transcript.get("Domain", "") if isinstance(transcript, dict) else ""
            parts = file_id.split("_")
            category = parts[-2] if len(parts) >= 2 else "unknown"

            # Store file with metadata (including speaker info)
            shard_files.append({
                'audio': fileobj.read(),
                'metadata': {
                    'text': text,
                    'file_id': file_id,
                    'category': category,
                    'domain': domain,
                    'speaker_id': SPEAKER_METADATA['speaker_id'],
                    'speaker_gender': SPEAKER_METADATA['speaker_gender'],
                    'speaker_age': SPEAKER_METADATA['speaker_age'],
                    'language': SPEAKER_METADATA['language'],
                }
            })

            total_files += 1

            # Create TAR shard when batch is full
            if len(shard_files) >= SHARD_SIZE:
                shard_num += 1
                create_and_upload_shard(
                    shard_files, shard_num, split_name,
                    temp_path, repo_id, api, total_files
                )
                shard_files = []

        # Upload remaining files
        if shard_files:
            shard_num += 1
            create_and_upload_shard(
                shard_files, shard_num, split_name,
                temp_path, repo_id, api, total_files
            )

    print(f"\n✓ Completed {split_name}: {total_files} files in {shard_num} shards")
    return total_files, shard_num


def create_and_upload_shard(shard_files, shard_num, split_name, temp_path, repo_id, api, total_files):
    """Create WebDataset TAR shard and upload"""
    print(f"Shard {shard_num}: Creating TAR with {len(shard_files)} files...")

    # Create TAR file
    tar_path = temp_path / f"{split_name}_{shard_num:05d}.tar"

    with tarfile.open(tar_path, 'w') as tar:
        for idx, item in enumerate(shard_files):
            # WebDataset format: files with same prefix
            prefix = f"{shard_num:05d}_{idx:06d}"

            # Add audio file - FIXED: Use .wav extension for WAV data!
            audio_info = tarfile.TarInfo(name=f"{prefix}.wav")
            audio_info.size = len(item['audio'])
            tar.addfile(audio_info, fileobj=__import__('io').BytesIO(item['audio']))

            # Add metadata JSON
            metadata_bytes = json.dumps(item['metadata'], ensure_ascii=False).encode('utf-8')
            metadata_info = tarfile.TarInfo(name=f"{prefix}.json")
            metadata_info.size = len(metadata_bytes)
            tar.addfile(metadata_info, fileobj=__import__('io').BytesIO(metadata_bytes))

    # Upload TAR shard
    tar_size_mb = tar_path.stat().st_size / 1024 / 1024
    print(f"  Uploading {tar_size_mb:.1f}MB TAR shard ({total_files} total files)...")

    api.upload_file(
        path_or_fileobj=str(tar_path),
        path_in_repo=f"data/{split_name}/{split_name}_{shard_num:05d}.tar",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add {split_name} shard {shard_num}"
    )

    print(f"  ✓ Shard {shard_num} uploaded\n")
    tar_path.unlink()


def main(repo_id: str, private: bool = False):
    print("\n" + "="*60)
    print("Gujarati TTS WebDataset Upload (FIXED)")
    print(f"Repository: {repo_id}")
    print(f"Shard size: {SHARD_SIZE} files per TAR")
    print(f"Speaker: {SPEAKER_METADATA['speaker_id']} ({SPEAKER_METADATA['speaker_gender']}, {SPEAKER_METADATA['speaker_age']})")
    print("="*60)

    # Create repo
    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        print(f"✓ Repository ready\n")
    except Exception as e:
        print(f"Note: {e}\n")

    api = HfApi()

    total_stats = {}
    for split_name, url in URLS.items():
        files, shards = process_split(url, split_name, repo_id, api)
        total_stats[split_name] = {"files": files, "shards": shards}

    # Create README with WebDataset loading instructions
    readme = f"""---
license: cc-by-4.0
task_categories:
- text-to-speech
language:
- gu
size_categories:
- 1K<n<10K
---

# Gujarati TTS Dataset

SPICOR Gujarati Female TTS dataset in WebDataset format.

## Dataset Details

- **Total Files**: {sum(s['files'] for s in total_stats.values())}
- **Duration**: ~33.6 hours
- **Speaker**: {SPEAKER_METADATA['speaker_id']} (Female, Age {SPEAKER_METADATA['speaker_age']})
- **Language**: Gujarati
- **Domains**: 19 domains (Agriculture, Entertainment, Finance, Health, Science, Sports, etc.)
- **Recording**: 48kHz, 24-bit, Studio quality

## Loading the Dataset

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("webdataset", data_dir="{repo_id}/resolve/main/data")

# Access audio and text
for sample in dataset["train"]:
    audio = sample["wav"]  # Audio bytes (WAV format)
    metadata = sample["json"]  # Metadata dict
    print(metadata["text"])  # Gujarati transcription
    print(metadata["speaker_id"])  # Speaker ID
    print(metadata["domain"])  # Domain
```

## Columns

Each sample contains:
- **audio** (`.wav` file): Raw WAV audio bytes
- **metadata** (`.json` file):
  - `text`: Gujarati transcription
  - `file_id`: Unique identifier
  - `category`: Category code (e.g., SPOR, AGRI)
  - `domain`: Full domain name
  - `speaker_id`: {SPEAKER_METADATA['speaker_id']}
  - `speaker_gender`: {SPEAKER_METADATA['speaker_gender']}
  - `speaker_age`: {SPEAKER_METADATA['speaker_age']}
  - `language`: gu (Gujarati)

## Splits

- **train**: {total_stats.get('train', {}).get('files', 0)} files in {total_stats.get('train', {}).get('shards', 0)} TAR shards
- **test**: {total_stats.get('test', {}).get('files', 0)} files in {total_stats.get('test', {}).get('shards', 0)} TAR shards

## Citation

```
@misc{{SPICOR_TTS_2.0_Corpus,
    Title = {{SPICOR TTS_2.0 Corpus - A 57+ hour domain-rich Gujarati TTS Corpus}},
    Authors = {{Abhayjeet Et al.}},
    Year = {{2025}}
}}
```

## License

CC-BY-4.0
"""

    # Upload README
    api.upload_file(
        path_or_fileobj=readme.encode('utf-8'),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"\n{'='*60}")
    print(f"✓ Complete!")
    for split, stats in total_stats.items():
        print(f"  {split}: {stats['files']} files in {stats['shards']} TAR shards")
    print(f"\nView at: https://huggingface.co/datasets/{repo_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    main(args.repo_id, args.private)
