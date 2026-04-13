# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# # VoiceBank Speech Enhancement: Dataset class
# 
# **Project Goal**: Build a deep learning model to denoise speech using the VoiceBank-DEMAND dataset.
# 
# **Notebook Overview**:
# -
# - PyTorch Dataset class loading with caching
# - Caching: persistent `.pkl` chunk cache to speed up reloading
# - Extract and save log-Mel spectrograms from clean and noisy audio for faster training
# - Model definition and training (next steps)
# 
# **Next Steps**:
# - Define data loaders (`train_loader`, `val_loader`)
# - Build and train the neural network model
# - Evaluate and visualize performance
# 
# **Author**: Jo  
# **Date**: Aug # !pip install torchaudio librosa soundfile matplotlib --quiet


# ## Dataset class
import os
import torchaudio
import torch
import librosa
import numpy as np
import soundfile as sf

import glob
import random
import pickle
from tqdm import tqdm


from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


import torch.nn as nn

class FeatureDataset(Dataset):
    """
    Enhanced dataset class for loading precomputed features
    Supports multiple feature types and data validation
    """
    
    def __init__(self, feature_dir, file_list, feature_type='logmel', normalize=True, augment=False, augment_prob=0.3):
        """
        Args:
            feature_dir (str): Directory containing features
            file_list (list): List of file IDs (without extensions)
            feature_type (str): Type of features ('logmel', 'stft', 'mfcc')
            normalize (bool): Apply normalization
            augment (bool): Apply data augmentation
            augment_prob (float): Probability of applying augmentation
        """
        self.feature_dir = feature_dir
        self.file_list = file_list
        self.feature_type = feature_type
        self.normalize = normalize
        self.augment = augment
        self.augment_prob = augment_prob
        
        # Validate dataset
        self._validate_dataset()
        
        # Compute normalization statistics if needed
        if self.normalize:
            self.stats = self._compute_normalization_stats()
    
    def _validate_dataset(self):
        """Validate that all required files exist"""
        missing_files = []
        
        for file_id in self.file_list:
            noisy_path = os.path.join(self.feature_dir, self.feature_type, 
                                    'noisy', f'{file_id}.npy')
            clean_path = os.path.join(self.feature_dir, self.feature_type, 
                                    'clean', f'{file_id}.npy')
            
            if not os.path.exists(noisy_path):
                missing_files.append(f"noisy/{file_id}.npy")
            if not os.path.exists(clean_path):
                missing_files.append(f"clean/{file_id}.npy")
        
        if missing_files:
            print(f"Warning: {len(missing_files)} missing files found")
            if len(missing_files) < 10:
                print("Missing files:", missing_files)
        
        # Remove files that don't have both noisy and clean versions
        valid_files = []
        for file_id in self.file_list:
            noisy_path = os.path.join(self.feature_dir, self.feature_type, 
                                    'noisy', f'{file_id}.npy')
            clean_path = os.path.join(self.feature_dir, self.feature_type, 
                                    'clean', f'{file_id}.npy')
            
            if os.path.exists(noisy_path) and os.path.exists(clean_path):
                valid_files.append(file_id)
        
        removed_count = len(self.file_list) - len(valid_files)
        if removed_count > 0:
            print(f"Removed {removed_count} invalid files")
        
        self.file_list = valid_files
        print(f"Validated dataset: {len(self.file_list)} valid samples")
    
    def _compute_normalization_stats(self, max_samples=500):
        """Compute normalization statistics from a subset of data"""
        #for script, link directly to file
        stats_file = "/kaggle/input/normalisation-stats/logmel/norm_stats_logmel.pkl"
        #stats_file = os.path.join(self.feature_type, 
                                 #f'norm_stats_{self.feature_type}.pkl')
        #stats_file = os.path.join(self.feature_dir, self.feature_type, 
                                 #f'final_stats.pkl')
        
        # Load cached stats if available
        if os.path.exists(stats_file):
            print(os.path.exists(stats_file))
            try:
                with open(stats_file, 'rb') as f:
                    stats = pickle.load(f)
                print(f"Loaded normalization stats from cache")
                return stats
            except:
                print("Failed to load cached stats, recomputing...")
        
                print("Computing normalization statistics...")
                # Sample files for statistics
                sample_files = random.sample(self.file_list, 
                                   min(max_samples, len(self.file_list)))
                all_noisy = []
                all_clean = []
                for file_id in tqdm(sample_files, desc="Computing stats"):
                    try:
                        noisy_feat = self._load_feature(file_id, 'noisy')
                        clean_feat = self._load_feature(file_id, 'clean')
                        all_noisy.append(noisy_feat.flatten())
                        all_clean.append(clean_feat.flatten())
                    except:
                        continue
                # Compute statistics
                all_noisy = np.concatenate(all_noisy)
                all_clean = np.concatenate(all_clean)
        
                stats = {'noisy_mean': np.mean(all_noisy), 
                         'noisy_std': np.std(all_noisy),
                         'clean_mean': np.mean(all_clean),
                         'clean_std': np.std(all_clean), 
                         'global_min': min(np.min(all_noisy), np.min(all_clean)), 
                         'global_max': max(np.max(all_noisy), np.max(all_clean))}
                # Cache the statistics
                os.makedirs(os.path.dirname(stats_file), exist_ok=True)
                with open(stats_file, 'wb') as f:
                    pickle.dump(stats, f)
        
                print(f"Cached normalization stats")
                print(f"   Noisy: μ={stats['noisy_mean']:.3f}, σ={stats['noisy_std']:.3f}")
                print(f"   Clean: μ={stats['clean_mean']:.3f}, σ={stats['clean_std']:.3f}")
        
                return stats
    
    def _load_feature(self, file_id, split):
        """Load a single feature file"""
        file_path = os.path.join(self.feature_dir, self.feature_type, 
                               split, f'{file_id}.npy')
        return np.load(file_path)
    
    def _normalize_feature(self, feature, split='noisy'):
        """Normalize feature using computed statistics"""
        if not self.normalize:
            return feature
        
        mean = self.stats[f'{split}_mean']
        std = self.stats[f'{split}_std']
        
        return (feature - mean) / (std + 1e-8)
    
    def _augment_feature(self, noisy_feat, clean_feat):
        """Apply feature-space augmentation"""
        if not self.augment or random.random() > self.augment_prob:
            return noisy_feat, clean_feat
        
        # Feature-space augmentations
        aug_type = random.choice(['noise', 'scale', 'shift', 'dropout'])
        
        if aug_type == 'noise':
            # Add noise to noisy features only
            noise_std = 0.05 * np.std(noisy_feat)
            noise = np.random.normal(0, noise_std, noisy_feat.shape)
            noisy_feat = noisy_feat + noise
            
        elif aug_type == 'scale':
            # Random scaling
            scale = random.uniform(0.9, 1.1)
            noisy_feat = noisy_feat * scale
            clean_feat = clean_feat * scale
            
        elif aug_type == 'shift':
            # Random DC shift
            shift = random.uniform(-0.1, 0.1)
            noisy_feat = noisy_feat + shift
            clean_feat = clean_feat + shift
            
        elif aug_type == 'dropout':
            # Feature dropout (set random features to 0)
            dropout_prob = 0.05
            mask = np.random.binomial(1, 1-dropout_prob, noisy_feat.shape)
            noisy_feat = noisy_feat * mask
        
        return noisy_feat, clean_feat
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        try:
            file_id = self.file_list[idx]
            
            # Load features
            noisy_feat = self._load_feature(file_id, 'noisy')
            clean_feat = self._load_feature(file_id, 'clean')
            
            # Apply augmentation
            noisy_feat, clean_feat = self._augment_feature(noisy_feat, clean_feat)
            
            # Normalize
            noisy_feat = self._normalize_feature(noisy_feat, 'noisy')
            clean_feat = self._normalize_feature(clean_feat, 'clean')
            
            # Convert to tensors
            noisy_tensor = torch.tensor(noisy_feat, dtype=torch.float32)
            clean_tensor = torch.tensor(clean_feat, dtype=torch.float32)
            
            return noisy_tensor, clean_tensor
            
        except Exception as e:
            print(f"Error loading {file_id}: {e}")
            # Return a zero tensor in case of error
            dummy_shape = (1, 64, 126)  # Default shape for logmel
            return (torch.zeros(dummy_shape, dtype=torch.float32),
                    torch.zeros(dummy_shape, dtype=torch.float32))

def get_file_list(feature_dir, feature_type='logmel'):
    """Get list of available feature files"""
    noisy_dir = os.path.join(feature_dir, feature_type, 'noisy')
    
    if not os.path.exists(noisy_dir):
        raise FileNotFoundError(f"Feature directory not found: {noisy_dir}")
    
    files = [f.replace('.npy', '') for f in os.listdir(noisy_dir) 
             if f.endswith('.npy')]
    
    return sorted(files)

def create_datasets(feature_dir, feature_type='logmel', val_ratio=0.1, 
                   test_ratio=0.1, random_state=42, normalize=True, 
                   augment_train=True):
    """
    Create train, validation, and test datasets
    
    Args:
        feature_dir (str): Directory containing features
        feature_type (str): Type of features to load
        val_ratio (float): Validation split ratio
        test_ratio (float): Test split ratio
        random_state (int): Random seed
        normalize (bool): Apply normalization
        augment_train (bool): Apply augmentation to training set
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    
    print(f"Creating datasets for {feature_type} features...")
    
    # Get file list
    file_list = get_file_list(feature_dir, feature_type)
    print(f"Found {len(file_list)} feature files")
    
    # Split into train/val/test
    train_files, temp_files = train_test_split(
        file_list, test_size=(val_ratio + test_ratio), 
        random_state=random_state
    )
    
    # Split temp into val and test
    val_files, test_files = train_test_split(
        temp_files, test_size=(test_ratio / (val_ratio + test_ratio)), 
        random_state=random_state
    )
    
    print(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    # Create datasets
    
    train_dataset = FeatureDataset(
        feature_dir=feature_dir,
        file_list=train_files, 
        feature_type=feature_type,      
        normalize=normalize,
        augment=augment_train,
        augment_prob=0.3
    )
    
    val_dataset = FeatureDataset(
        feature_dir=feature_dir,
        file_list=val_files, 
        feature_type=feature_type,      
        normalize=normalize, 
        augment=False
    )
    
    test_dataset = FeatureDataset(
        feature_dir=feature_dir,
        file_list=test_files, 
        feature_type=feature_type,     
        normalize=normalize, 
        augment=False
    )
    
    return train_dataset, val_dataset, test_dataset

def create_dataloaders(feature_dir, feature_type='logmel', batch_size=16, 
                      val_ratio=0.1, test_ratio=0.1, random_state=42, 
                      num_workers=2, normalize=True, augment_train=True):
    """
    Create DataLoaders for training, validation, and testing
    
    Args:
        feature_dir (str): Directory containing features
        feature_type (str): Type of features to load
        batch_size (int): Batch size for DataLoaders
        val_ratio (float): Validation split ratio
        test_ratio (float): Test split ratio
        random_state (int): Random seed
        num_workers (int): Number of workers for DataLoader
        normalize (bool): Apply normalization
        augment_train (bool): Apply augmentation to training set
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        feature_dir, feature_type, val_ratio, test_ratio, 
        random_state, normalize, augment_train
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Created DataLoaders:")
    print(f"   Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"   Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"   Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader



# from voicebank_dataset import VoiceBankDataset  # custom dataset class saved as .py file
class VoiceBankDataset(Dataset):
    """VoiceBank Dataset with improved caching and augmentation"""
    
    def __init__(self, clean_dir, noisy_dir, sample_rate=16000, min_rms = 0.01, augment_prob = 0.5, segment_duration=1.0, cache_dir=None, use_cache=False,augment=True):
        self.clean_files = sorted(glob.glob(f"{clean_dir}/**/*.wav", recursive=True))
        self.noisy_files = sorted(glob.glob(f"{noisy_dir}/**/*.wav", recursive=True))
        assert len(self.clean_files) == len(self.noisy_files), f"Mismatch: {len(self.clean_files)} clean files vs {len(self.noisy_files)} noisy files"
        self.sr = sample_rate
        self.seg_len = int(sample_rate * segment_duration)
        self.min_rms = min_rms
        self.augment_prob = augment_prob
        self.augment = augment
        self.cache_dir = cache_dir
        self.use_cache = use_cache     

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)  
            self.cache_file = os.path.join(cache_dir, "chunks.pkl")
        else:
            self.cache_file = None

        #Create or load chunks

        self._initialize_chunks()
        
    def _initialize_chunks(self):
        """Initialize dataset chunks with improved caching"""
        if self.use_cache and self.cache_file and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                    self.chunks = cache_data['chunks']
                    print(f"Loaded {len(self.chunks)} cached chunks from {self.cache_file}")
                    return
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Cache file corrupted: {e}. Rebuilding cache...")
        
        # Create chunks from scratch
        print("Processing audio files...")
        self.chunks = self._preprocess_chunks()
        
        # Save to cache
        if self.cache_file:
            cache_data = {
                'chunks': self.chunks,
                'sample_rate': self.sr,
                'segment_duration': self.seg_len / self.sr,
                'min_rms': self.min_rms
            }
            with open(self.cache_file, "wb") as f:
                pickle.dump(cache_data, f)
            print(f"Saved {len(self.chunks)} chunks to cache")
       

        
        

    def _rms(self, signal):
        return np.sqrt(np.mean(signal ** 2))

    def _preprocess_chunks(self):
        """Process audio files into fixed-length chunks"""
        chunks = []
        for clean_path, noisy_path in tqdm(zip(self.clean_files, self.noisy_files), 
                                          total=len(self.clean_files), 
                                          desc="Processing audio files"):
            
            try:
                #load audio files
                clean, _ = librosa.load(clean_path, sr=self.sr)
                noisy, _ = librosa.load(noisy_path, sr=self.sr)
                #normalise audio files
                clean = librosa.util.normalize(clean)
                noisy = librosa.util.normalize(noisy)
                # Ensure same length
                min_len = min(len(clean), len(noisy)) # also total number of samples
                clean = clean[:min_len]
                noisy = noisy[:min_len]

                # Chunk into fixed 1-sec segments
            
                for start in range(0, min_len, self.seg_len):
                    end = start + self.seg_len
                    clean_chunk = clean[start:end]
                    noisy_chunk = noisy[start:end]

                    # Pad if needed
                    if len(clean_chunk) < self.seg_len:
                        clean_chunk = librosa.util.fix_length(clean_chunk, size = self.seg_len)
                        noisy_chunk = librosa.util.fix_length(noisy_chunk, size = self.seg_len)

                    # Filter out low-energy segments
                    if self._rms(clean_chunk) >= self.min_rms:
                        chunks.append((noisy_chunk.astype(np.float32), clean_chunk.astype(np.float32)))
            except Exception as e:
                print(f"Error processing {clean_path}: {e}")
        print(f"Created {len(chunks)} audio chunks")
        return chunks

    def _augment(self, noisy, clean):

        if not self.augment or random.random() > self.augment_prob:
            return noisy, clean
        aug_type = random.choice(['pitch', 'stretch', 'noise', 'gain'])

        try:
        
            if aug_type == 'pitch':
                # pitch shift
                n_steps = random.uniform(-1.5, 1.5)
                noisy = librosa.effects.pitch_shift(noisy, sr=self.sr, n_steps=n_steps)
                clean = librosa.effects.pitch_shift(clean, sr=self.sr, n_steps=n_steps)
            elif aug_type == 'stretch':
                # time stretch
                rate = random.uniform(0.85, 1.15)
                noisy = librosa.effects.time_stretch(noisy, rate=rate)
                clean = librosa.effects.time_stretch(clean, rate=rate)
                # Fix length back to segment
                noisy = librosa.util.fix_length(noisy, size = self.seg_len)
                clean = librosa.util.fix_length(clean, size = self.seg_len)
            elif aug_type == 'noise':
                # add noise to noisy signal only
                noise_level = random.uniform(0.005, 0.02)
                noise = np.random.normal(0, noise_level, size=noisy.shape)
                noisy = noisy + noise
                noisy = np.clip(noisy, -1.0, 1.0)
            elif aug_type == 'gain':
                # Random gain adjustment
                gain = random.uniform(0.7, 1.3)
                noisy = noisy * gain
                clean = clean * gain
                noisy = np.clip(noisy, -1.0, 1.0)
                clean = np.clip(clean, -1.0, 1.0)
        except Exception as e:
            # if augmentation fails return original signal
            pass
        return noisy, clean

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        noisy, clean = self.chunks[idx]
        # Apply augmentation
        noisy, clean = self._augment(noisy, clean)
        # Convert to tensors and add channel dimension
        return (torch.tensor(noisy, dtype=torch.float32).unsqueeze(0),
                torch.tensor(clean, dtype=torch.float32).unsqueeze(0))

                    
          

