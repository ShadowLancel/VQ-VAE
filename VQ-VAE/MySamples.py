import librosa
import soundfile as sf
import os
import numpy as np

# Define paths
source_dir = r'D:\output_flac'
target_dir = r'E:\C418_all'
target_sr = 44100
duration = 4

# Create target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Iterate through audio files in source directory
for filename in os.listdir(source_dir):
    if filename.endswith('.flac'):
        filepath = os.path.join(source_dir, filename)

        try:
            # Load audio file with librosa
            y, sr = librosa.load(filepath, sr=target_sr)

            # Calculate the number of samples for 4 seconds
            num_samples = int(target_sr * duration)

            # Handle audio files shorter than 4 seconds
            if len(y) < num_samples:
              print(f"File {filename} is shorter than 4 seconds, padding with zeros.")
              y = np.pad(y, (0, num_samples - len(y)), mode='constant')
            # Extract 4-second samples
            for i in range(0, len(y) - num_samples + 1, num_samples):
                sample = y[i : i + num_samples]
                sample_filename = os.path.splitext(filename)[0] + f'_{i//num_samples}.wav'
                sample_filepath = os.path.join(target_dir, sample_filename)
                sf.write(sample_filepath, sample, target_sr)

        except Exception as e:
            print(f"Error processing file {filename}: {e}")