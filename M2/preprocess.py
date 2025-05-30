import os
import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import json

class CatMeowDatasetPreprocessor:
    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.sr = 22050  # Sample rate for HiFi-GAN
        self.n_fft = 1024
        self.hop_length = 256
        self.n_mels = 80
        
        # Define emotion categories based on typical cat behavior
        self.emotion_categories = {
            'fear': 0,
            'joy': 1, 
            'anticipation/demanding': 2,
            'neutral': 3
        }
        
    def extract_mel_spectrogram(self, audio_path):
        """Extract mel-spectrogram from audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # Ensure minimum length (pad if too short)
            min_length = self.sr * 1  # 1 second minimum
            if len(y) < min_length:
                y = np.pad(y, (0, min_length - len(y)), mode='constant')
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=0,
                fmax=8000
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to [-1, 1]
            mel_spec_norm = 2 * (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min()) - 1
            
            return mel_spec_norm
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def prepare_dataset(self):
        """
        Prepare the dataset by extracting features and organizing by emotion
        
        Expected directory structure:
        dataset_path/
        ├── hungry/
        │   ├── meow1.wav
        │   ├── meow2.wav
        ├── happy/
        │   ├── meow1.wav
        │   ├── meow2.wav
        └── ...
        """
        dataset_info = []
        mel_spectrograms = []
        emotion_labels = []
        
        print("Processing cat meow dataset...")
        
        for emotion_folder in os.listdir(self.dataset_path):
            emotion_path = os.path.join(self.dataset_path, emotion_folder)
            
            if not os.path.isdir(emotion_path):
                continue
                
            if emotion_folder not in self.emotion_categories:
                print(f"Warning: Unknown emotion category '{emotion_folder}', skipping...")
                continue
            
            emotion_id = self.emotion_categories[emotion_folder]
            print(f"Processing {emotion_folder} meows...")
            
            audio_files = [f for f in os.listdir(emotion_path) if f.endswith(('.wav', '.mp3', '.flac'))]
            
            for audio_file in audio_files:
                audio_path = os.path.join(emotion_path, audio_file)
                
                # Extract mel-spectrogram
                mel_spec = self.extract_mel_spectrogram(audio_path)
                
                if mel_spec is not None:
                    mel_spectrograms.append(mel_spec)
                    emotion_labels.append(emotion_id)
                    
                    dataset_info.append({
                        'file_path': audio_path,
                        'emotion': emotion_folder,
                        'emotion_id': emotion_id,
                        'mel_shape': mel_spec.shape
                    })
        
        # Save processed data
        os.makedirs(self.output_path, exist_ok=True)
        
        # Save mel-spectrograms
        np.save(os.path.join(self.output_path, 'mel_spectrograms.npy'), mel_spectrograms)
        np.save(os.path.join(self.output_path, 'emotion_labels.npy'), emotion_labels)
        
        # Save dataset info
        with open(os.path.join(self.output_path, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Save emotion mapping
        with open(os.path.join(self.output_path, 'emotion_mapping.json'), 'w') as f:
            json.dump(self.emotion_categories, f, indent=2)
        
        print(f"Dataset prepared successfully!")
        print(f"Total samples: {len(mel_spectrograms)}")
        print(f"Emotion distribution:")
        
        emotion_counts = {}
        for label in emotion_labels:
            emotion_name = [k for k, v in self.emotion_categories.items() if v == label][0]
            emotion_counts[emotion_name] = emotion_counts.get(emotion_name, 0) + 1
        
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count} samples")
        
        return mel_spectrograms, emotion_labels

class CatMeowDataset(Dataset):
    """PyTorch Dataset for cat meow training"""
    
    def __init__(self, mel_spectrograms, emotion_labels, max_length=None):
        self.mel_spectrograms = mel_spectrograms
        self.emotion_labels = emotion_labels
        self.max_length = max_length
        
        # Find maximum length for padding
        if max_length is None:
            self.max_length = max([spec.shape[1] for spec in mel_spectrograms])
        
    def __len__(self):
        return len(self.mel_spectrograms)
    
    def __getitem__(self, idx):
        mel_spec = self.mel_spectrograms[idx]
        emotion_label = self.emotion_labels[idx]
        
        # Pad or truncate to fixed length
        mel_spec = self._pad_or_truncate(mel_spec)
        
        return {
            'mel_spectrogram': torch.FloatTensor(mel_spec),
            'emotion_label': torch.LongTensor([emotion_label])
        }
    
    def _pad_or_truncate(self, mel_spec):
        """Pad or truncate mel-spectrogram to fixed length"""
        current_length = mel_spec.shape[1]
        
        if current_length > self.max_length:
            # Truncate
            start_idx = np.random.randint(0, current_length - self.max_length + 1)
            mel_spec = mel_spec[:, start_idx:start_idx + self.max_length]
        elif current_length < self.max_length:
            # Pad
            pad_length = self.max_length - current_length
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_length)), mode='constant', constant_values=0)
        
        return mel_spec

# Text dataset for intent classifier
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def create_text_emotion_dataset(csv_file_path):
    """Load existing text-emotion pairs from CSV file"""
    try:
        # Load your existing dataset
        df = pd.read_csv(csv_file_path)
        
        # Rename columns to match expected format
        df = df.rename(columns={'intent': 'emotion'})
        
        # Use predefined emotion mapping to match preprocessing
        emotion_mapping = {
            'fear': 0,
            'joy': 1,
            'anticipation/demanding': 2,
            'neutral': 3
        }
        
        # Check if all emotions in dataset are in mapping
        unique_emotions = df['emotion'].unique()
        unmapped_emotions = [e for e in unique_emotions if e not in emotion_mapping]
        if unmapped_emotions:
            print(f"Warning: Found unmapped emotions: {unmapped_emotions}")
            print("These will be assigned NaN emotion_id values")
        
        # Add emotion IDs
        df['emotion_id'] = df['emotion'].map(emotion_mapping)
        
        # Remove rows with unmapped emotions (NaN emotion_id)
        original_len = len(df)
        df = df.dropna(subset=['emotion_id'])
        df['emotion_id'] = df['emotion_id'].astype(int)
        
        if len(df) < original_len:
            print(f"Removed {original_len - len(df)} rows with unmapped emotions")
        
        print(f"Loaded {len(df)} text-emotion pairs")
        print(f"Available emotions: {list(emotion_mapping.keys())}")
        
        return df, emotion_mapping
        
    except FileNotFoundError:
        print(f"Error: Could not find file {csv_file_path}")
        return None, None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

def create_text_emotion_dataset_combined(csv_file_path=None):
    """Create text-emotion dataset with option to load from file or use hardcoded data"""
    
    if csv_file_path and os.path.exists(csv_file_path):
        # Load from existing CSV
        return create_text_emotion_dataset(csv_file_path)
    
    else:
        # Fallback to original hardcoded data
        text_data = [
            # Fear/Scared
            ("I'm scared", "fear"),
            ("Help me", "fear"),
            ("I'm afraid", "fear"),
            ("Something's wrong", "fear"),
            ("I don't feel safe", "fear"),
            ("There's danger", "fear"),
            ("I'm worried", "fear"),
            ("Hide me", "fear"),
            
            # Joy/Happy
            ("I'm so happy", "joy"),
            ("Let's play", "joy"),
            ("I love you", "joy"),
            ("This is fun", "joy"),
            ("I'm excited", "joy"),
            ("Play with me", "joy"),
            ("I'm feeling great", "joy"),
            ("Time to run around", "joy"),
            
            # Anticipation/Demanding
            ("I'm hungry", "anticipation/demanding"),
            ("Feed me now", "anticipation/demanding"),
            ("Where's my food", "anticipation/demanding"),
            ("I want to eat", "anticipation/demanding"),
            ("Give me treats", "anticipation/demanding"),
            ("My bowl is empty", "anticipation/demanding"),
            ("I'm starving", "anticipation/demanding"),
            ("Dinner time", "anticipation/demanding"),
            ("Look at me", "anticipation/demanding"),
            ("Pet me please", "anticipation/demanding"),
            ("I want attention", "anticipation/demanding"),
            ("Notice me", "anticipation/demanding"),
            ("Come here", "anticipation/demanding"),
            ("I need cuddles", "anticipation/demanding"),
            ("Don't ignore me", "anticipation/demanding"),
            ("Spend time with me", "anticipation/demanding"),
            
            # Neutral
            ("I'm comfortable", "neutral"),
            ("Life is good", "neutral"),
            ("I'm relaxed", "neutral"),
            ("This is nice", "neutral"),
            ("I'm at peace", "neutral"),
            ("Everything's perfect", "neutral"),
            ("I'm satisfied", "neutral"),
            ("So cozy", "neutral"),
        ]
        
        # Create DataFrame
        df = pd.DataFrame(text_data, columns=['text', 'emotion'])
        
        # Add emotion IDs using predefined mapping
        emotion_mapping = {
            'fear': 0,
            'joy': 1,
            'anticipation/demanding': 2,
            'neutral': 3
        }
        
        df['emotion_id'] = df['emotion'].map(emotion_mapping)
        return df, emotion_mapping

def prepare_training_data(dataset_path, output_path, text_csv_path=None):
    """Complete data preparation pipeline"""
    print("Step 1: Preparing cat meow audio dataset...")
    preprocessor = CatMeowDatasetPreprocessor(dataset_path, output_path)
    mel_specs, emotion_labels = preprocessor.prepare_dataset()
    
    print("\nStep 2: Creating text-emotion dataset...")
    text_df, emotion_mapping = create_text_emotion_dataset_combined(text_csv_path)
    
    if text_df is not None:
        # Save the processed dataset
        output_csv_path = os.path.join(output_path, 'text_emotion_dataset.csv')
        text_df.to_csv(output_csv_path, index=False)
        print(f"Text-emotion dataset saved to: {output_csv_path}")
        
        # Also save emotion mapping for later use
        mapping_path = os.path.join(output_path, 'emotion_mapping.txt')
        with open(mapping_path, 'w') as f:
            for emotion, idx in emotion_mapping.items():
                f.write(f"{emotion}: {idx}\n")
    
    print("\nStep 3: Creating train/validation splits...")
    # Split audio data
    mel_train, mel_val, emotion_train, emotion_val = train_test_split(
        mel_specs, emotion_labels, test_size=0.2, random_state=42, stratify=emotion_labels
    )
    
    # Split text data if available
    if text_df is not None:
        text_train, text_val = train_test_split(
            text_df, test_size=0.2, random_state=42, stratify=text_df['emotion']
        )
        
        # Save text splits
        text_train.to_csv(os.path.join(output_path, 'text_train.csv'), index=False)
        text_val.to_csv(os.path.join(output_path, 'text_val.csv'), index=False)
    
    return mel_train, mel_val, emotion_train, emotion_val, text_df
# Usage example:
# For using your existing CSV file:
# prepare_training_data(dataset_path, output_path, text_csv_path="path/to/your/dataset.csv")

# For using hardcoded data (original behavior):
# prepare_training_data(dataset_path, output_path)

if __name__ == "__main__":
    # Adjust these paths to your dataset
    dataset_path = "/content/drive/MyDrive/CatMeowsDataset/raw_audio"  # Your organized cat meow folders
    output_path = "/content/drive/MyDrive/CatMeowsDataset/processed"
    
    prepare_training_data(dataset_path, output_path)
