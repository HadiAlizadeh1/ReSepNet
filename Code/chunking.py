# In the Name of Allah
# Script for padding or chunking audio files to a fixed duration (in seconds).
# - Pads audio shorter than target_length seconds
# - Splits audio longer than target_length seconds into equal chunks
# Created: 2024 by Hadi Alizadeh

import os
import argparse
import numpy as np
import soundfile as sf


def process_audio_file(filepath, target_length_sec, sample_rate):
    """Pad or chunk a single .wav file."""
    audio_data, sr = sf.read(filepath)

    if sr != sample_rate:
        print(f"⚠️ Warning: {filepath} has sample rate {sr}, expected {sample_rate}")

    target_samples = int(target_length_sec * sample_rate)
    num_samples = len(audio_data)

    # Case 1: Pad if shorter than target duration
    if num_samples < target_samples:
        padded_audio = np.concatenate([audio_data, np.zeros(target_samples - num_samples)])
        sf.write(filepath, padded_audio, sample_rate)
        print(f"→ Padded: {filepath}")
        return

    # Case 2: Chunk if longer than target duration
    if num_samples > target_samples:
        os.remove(filepath)
        num_chunks = num_samples // target_samples
        remainder = num_samples % target_samples

        # Write each full chunk
        for i in range(num_chunks):
            start = i * target_samples
            end = start + target_samples
            chunk = audio_data[start:end]
            out_path = f"{filepath[:-4]}_{i+1}.wav"
            sf.write(out_path, chunk, sample_rate)
            print(f"→ Chunk saved: {out_path}")

        # Handle remainder (if longer than half a chunk)
        if remainder > target_samples // 2:
            tail = audio_data[-remainder:]
            tail = np.concatenate([tail, np.zeros(target_samples - len(tail))])
            out_path = f"{filepath[:-4]}_extra.wav"
            sf.write(out_path, tail, sample_rate)
            print(f"→ Extra chunk saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pad or chunk audio files to a fixed length (in seconds)."
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing speaker folders with .wav files."
    )
    parser.add_argument(
        "--target_length",
        type=float,
        default=4.0,
        help="Target audio length in seconds (default: 4.0)."
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=8000,
        help="Sample rate of the audio files (default: 8000 Hz)."
    )

    args = parser.parse_args()

    print("\n=== Audio Chunking Started ===")
    print(f"Base Directory : {args.base_dir}")
    print(f"Target Length  : {args.target_length} sec")
    print(f"Sample Rate    : {args.sample_rate} Hz\n")

    for speaker_folder in os.listdir(args.base_dir):
        speaker_path = os.path.join(args.base_dir, speaker_folder)
        if not os.path.isdir(speaker_path):
            continue

        for filename in os.listdir(speaker_path):
            if filename.endswith(".wav"):
                filepath = os.path.join(speaker_path, filename)
                process_audio_file(filepath, args.target_length, args.sample_rate)

    print("\n=== Processing Complete ===")


if __name__ == "__main__":
    main()
