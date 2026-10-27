"""
Dataset utilities for the SilentWear EMG silent-speech dataset
(Spacone et al., "SilentWear: an Ultra-Low Power Wearable System for
EMG-based Silent Speech Recognition").

The public dataset is hosted on HuggingFace:
    https://huggingface.co/datasets/PulpBio/SilentWear

This module supports two ways of getting data in:
  1. Directly from the HuggingFace `datasets` library (recommended).
  2. From local .npy files you've pre-extracted, shaped as:
        X: (N, 14, T)  float32   -- 14 EMG channels, T time samples
        y: (N,)        int64     -- 9 classes (0=rest, 1..8=commands)

Signal preprocessing (per the paper, Sec. III-C):
  - 4th-order zero-phase Butterworth bandpass filter, cutoff 20–200 Hz
  - sampling rate: 500 Hz

Commands (label order used throughout, index 0 is 'rest'):
    0: rest, 1: up, 2: down, 3: left, 4: right,
    5: forward, 6: backward, 7: start, 8: stop
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, iirnotch
from typing import Optional, Tuple, List

LABELS = ["rest", "up", "down", "left", "right", "forward", "backward", "start", "stop"]
NUM_CLASSES = len(LABELS)
NUM_CHANNELS = 14
SAMPLING_RATE_HZ = 500


def preprocess_emg(x: np.ndarray,
                   fs: int = SAMPLING_RATE_HZ,
                   lowcut: float = 20.0,
                   highcut: float = 200.0,
                   order: int = 4) -> np.ndarray:
    """Apply bandpass filter 20–200 Hz (4th‑order Butterworth)."""
    x = np.asarray(x, dtype=np.float64)
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    x = filtfilt(b, a, x, axis=-1)
    return x.astype(np.float32)


def window_ms_to_samples(window_ms: int, fs: int = SAMPLING_RATE_HZ) -> int:
    """Convert a window length in ms (e.g. 800, 1400) to a sample count."""
    return int(round(window_ms * fs / 1000.0))


class SilentWearDataset(Dataset):
    """Windowed EMG dataset ready for SpeechNet.

    Each item is (x, y):
        x: FloatTensor (1, 14, T)  -- unsqueezed channel dim for Conv2d
        y: LongTensor  scalar      -- class index in [0, 8]

    Parameters
    ----------
    X : np.ndarray, shape (N, 14, T_raw)
        Raw or already-preprocessed EMG windows.
    y : np.ndarray, shape (N,)
        Integer class labels.
    window_ms : int
        Target window length in ms (used only to crop/pad T_raw to T).
    apply_preprocessing : bool
        If True, run `preprocess_emg` on the fly. Set False if X was
        already filtered upstream (recommended for speed: filter once,
        cache to disk, then set this False).
    normalize : bool
        If True, z-score normalize each channel per-window.
    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 window_ms: int = 1400,
                 apply_preprocessing: bool = False,
                 normalize: bool = True,
                 num_channels: Optional[int] = None):
        if num_channels is None:
            num_channels = NUM_CHANNELS
        assert X.ndim == 3 and X.shape[1] == num_channels, \
            f"expected X shape (N, {num_channels}, T), got {X.shape}"
        assert len(X) == len(y)

        self.T = window_ms_to_samples(window_ms)
        self.apply_preprocessing = apply_preprocessing
        self.normalize = normalize

        X = self._fit_length(X, self.T)
        self.X = X.astype(np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    @staticmethod
    def _fit_length(X: np.ndarray, T: int) -> np.ndarray:
        """Center-crop or zero-pad the time axis to exactly T samples."""
        N, C, T_raw = X.shape
        if T_raw == T:
            return X
        if T_raw > T:
            start = (T_raw - T) // 2
            return X[:, :, start:start + T]
        pad_total = T - T_raw
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(X, ((0, 0), (0, 0), (pad_left, pad_right)), mode="constant")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        if self.apply_preprocessing:
            x = preprocess_emg(x)
        if self.normalize:
            mean = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True) + 1e-6
            x = (x - mean) / std
        x = torch.from_numpy(x).unsqueeze(0)  # (1, 14, T)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

    @classmethod
    def from_npy(cls, x_path: str, y_path: str, num_channels=None, **kwargs):
        X = np.load(x_path)
        y = np.load(y_path)
        return cls(X, y, num_channels=num_channels, **kwargs)

def make_loaders(train_ds: Dataset,
                 val_ds: Dataset,
                 batch_size: int = 32,
                 num_workers: int = 2):
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=False,
                               pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False,
                             pin_memory=torch.cuda.is_available())
    return train_loader, val_loader