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

def prepare_training_data(dataset_path, output_path):
    """Complete data preparation pipeline"""
    
    print("Step 1: Preparing cat meow audio dataset...")
    preprocessor = CatMeowDatasetPreprocessor(dataset_path, output_path)
    mel_specs, emotion_labels = preprocessor.prepare_dataset()
    
    print("\nStep 2: Creating text-emotion dataset...")
    text_df = create_text_emotion_dataset()
    text_df.to_csv(os.path.join(output_path, 'text_emotion_dataset.csv'), index=False)
    
    print("\nStep 3: Creating train/validation splits...")
    
    # Split audio data
    mel_train, mel_val, emotion_train, emotion_val = train_test_split(
        mel_specs, emotion_labels, test_size=0.2, random_state=42, stratify=emotion_labels
    )
    
    # Save splits
    np.save(os.path.join(output_path, 'mel_train.npy'), mel_train)
    np.save(os.path.join(output_path, 'mel_val.npy'), mel_val)
    np.save(os.path.join(output_path, 'emotion_train.npy'), emotion_train)
    np.save(os.path.join(output_path, 'emotion_val.npy'), emotion_val)
    
    # Split text data
    text_train, text_val = train_test_split(text_df, test_size=0.2, random_state=42, stratify=text_df['emotion'])
    text_train.to_csv(os.path.join(output_path, 'text_train.csv'), index=False)
    text_val.to_csv(os.path.join(output_path, 'text_val.csv'), index=False)
    
    print("Data preparation completed!")
    print(f"Training samples: {len(mel_train)} audio, {len(text_train)} text")
    print(f"Validation samples: {len(mel_val)} audio, {len(text_val)} text")

# Usage example
if __name__ == "__main__":
    # Adjust these paths to your dataset
    dataset_path = "/content/drive/MyDrive/CatMeowsDataset/raw_audio"  # Your organized cat meow folders
    output_path = "/content/drive/MyDrive/CatMeowsDataset/processed"
    
    prepare_training_data(dataset_path, output_path)
