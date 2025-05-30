import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class ResidualBlock(nn.Module):
    """Residual block for generator"""
    
    def __init__(self, channels, kernel_size=3, dilation=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=dilation, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=dilation, dilation=dilation
        )
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        return F.relu(out + residual)

class ConditionalMelGANGenerator(nn.Module):
    """Conditional MelGAN Generator for emotion-conditioned mel-spectrogram generation"""
    
    def __init__(self, num_emotions=4, noise_dim=100, mel_dim=80, seq_len=128):
        super(ConditionalMelGANGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.mel_dim = mel_dim
        self.seq_len = seq_len
        self.emotion_dim = 128
        
        # Emotion embedding
        self.emotion_embedding = nn.Embedding(num_emotions, self.emotion_dim)
        
        # Initial projection
        self.initial_conv = nn.Conv1d(
            noise_dim + self.emotion_dim, 512, kernel_size=7, padding=3
        )
        
        # Upsampling blocks
        self.upsample_blocks = nn.ModuleList([
            # 512 -> 256
            nn.Sequential(
                nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(1, 256),
                nn.ReLU(),
                ResidualBlock(256, dilation=1),
                ResidualBlock(256, dilation=3),
            ),
            # 256 -> 128  
            nn.Sequential(
                nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(1, 128),
                nn.ReLU(),
                ResidualBlock(128, dilation=1),
                ResidualBlock(128, dilation=3),
            ),
            # 128 -> 64
            nn.Sequential(
                nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(1, 64),
                nn.ReLU(),
                ResidualBlock(64, dilation=1),
                ResidualBlock(64, dilation=3),
            ),
        ])
        
        # Final convolution to mel dimensions
        self.final_conv = nn.Conv1d(64, mel_dim, kernel_size=7, padding=3)
        
    def forward(self, noise, emotion_ids):
        batch_size = noise.shape[0]
        
        # Get emotion embeddings
        emotion_emb = self.emotion_embedding(emotion_ids)  # [batch_size, emotion_dim]
        
        # Expand emotion embedding to match noise sequence length
        emotion_emb = emotion_emb.unsqueeze(2).repeat(1, 1, noise.shape[2])  # [batch_size, emotion_dim, seq_len]
        
        # Concatenate noise and emotion
        x = torch.cat([noise, emotion_emb], dim=1)  # [batch_size, noise_dim + emotion_dim, seq_len]
        
        # Initial convolution
        x = F.relu(self.initial_conv(x))
        
        # Upsampling blocks
        for block in self.upsample_blocks:
            x = block(x)
        
        # Final convolution with tanh activation
        mel_spec = torch.tanh(self.final_conv(x))
        
        return mel_spec

class MelGANDiscriminator(nn.Module):
    """Multi-scale discriminator for MelGAN"""
    
    def __init__(self, num_emotions=7, mel_dim=80):
        super(MelGANDiscriminator, self).__init__()
        
        # Emotion embedding for conditional discrimination
        self.emotion_embedding = nn.Embedding(num_emotions, 64)
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(mel_dim + 64, 128, kernel_size=15, stride=1, padding=7),
            nn.Conv1d(128, 256, kernel_
