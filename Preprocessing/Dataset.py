import os
import re
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define the path to the dataset
DATASET_PATH = 'dataset'  # Replace with your actual path

# Regular expression pattern to parse filenames
FILENAME_PATTERN = re.compile(
    r'^(?P<context>[BFI])_(?P<cat_id>\d{5})_(?P<breed>MC|EU)_(?P<sex>FI|FN|MI|MN)_(?P<owner_id>\d{5})_(?P<session>[123])(?P<vocalization>\d{2})\.wav$'
)

# Function to parse metadata from filename
def parse_filename(filename):
    match = FILENAME_PATTERN.match(filename)
    if match:
        return match.groupdict()
    else:
        return None

# Function to trim silence from audio
def trim_silence(audio, sr, top_db=20):
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio

# Function to extract MFCC features
def extract_mfcc(audio, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs

# Lists to store metadata and features
metadata_list = []
features_list = []

# Process each .wav file in the dataset
for root, _, files in os.walk(DATASET_PATH):
    for file in tqdm(files):
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            metadata = parse_filename(file)
            if metadata:
                try:
                    # Load audio file
                    audio, sr = librosa.load(file_path, sr=8000)
                    
                    # Trim silence
                    audio_trimmed = trim_silence(audio, sr)
                    
                    # Extract MFCC features
                    mfcc = extract_mfcc(audio_trimmed, sr)
                    
                    # Store metadata and features
                    metadata_entry = metadata.copy()
                    metadata_entry['file_path'] = file_path
                    metadata_list.append(metadata_entry)
                    features_list.append(mfcc)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            else:
                print(f"Filename does not match pattern: {file}")

# Convert metadata to DataFrame
metadata_df = pd.DataFrame(metadata_list)

# Save metadata to CSV
metadata_df.to_csv('catmeows_metadata.csv', index=False)

# Save features to NumPy file
np.save('catmeows_mfcc_features.npy', features_list)

print("Preprocessing completed successfully.")
