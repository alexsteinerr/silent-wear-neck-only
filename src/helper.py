import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.data.dataset import SilentWearDataset
from typing import Tuple, Optional

def load_split(
    x_path: str,
    y_path: str,
    window_ms: int,
    val_fraction: float = 0.15,
    seed: int = 0,
    meta_path: str = None,
    split_mode: str = "random",
    held_out_session: int = None,
    held_out_subject: str = None,
    channel_indices: Optional[list] = None,
) -> Tuple[SilentWearDataset, SilentWearDataset]:
    """
    Load X and y, optionally slice channels, then split into train/val.
    """
    X = np.load(x_path)
    y = np.load(y_path)
    print(f"\nDataset: X {X.shape}, y {y.shape}")

    if channel_indices is not None:
        X = X[:, channel_indices, :]
        print(f"Selected {len(channel_indices)} channels: {channel_indices}")
    else:
        channel_indices = list(range(X.shape[1]))

    # Perform split
    rng = np.random.RandomState(seed)
    n = len(y)

    if split_mode == "inter_session":
        if meta_path is None or held_out_session is None:
            raise ValueError("inter_session requires --meta_path and --held_out_session")
        
        meta = np.load(meta_path)
        sessions = meta["session"]
        val_mask = sessions == held_out_session
        train_mask = ~val_mask

        if not val_mask.any():
            raise ValueError(f"No examples for held_out_session={held_out_session}")

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        print(f"Inter-session split: train {len(train_idx)}, val {len(val_idx)}, held-out session {held_out_session}")

    elif split_mode == "inter_subject":
        if meta_path is None or held_out_subject is None:
            raise ValueError("inter_subject requires --meta_path and --held_out_subject")
        
        meta = np.load(meta_path)
        subjects = meta["subject"]

        val_mask = subjects == held_out_subject
        train_mask = ~val_mask
        if not val_mask.any():
            raise ValueError(f"No examples for held_out_subject={held_out_subject}")
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        print(f"Inter-subject split: train {len(train_idx)}, val {len(val_idx)}, held-out subject {held_out_subject}")

    else:  # random
        idx = rng.permutation(n)
        n_val = int(n * val_fraction)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        print(f"Random split: train {len(train_idx)}, val {len(val_idx)}")

    train_ds = SilentWearDataset(
        X[train_idx], y[train_idx],
        window_ms=window_ms,
        apply_preprocessing=False,
        normalize=True,
        num_channels=len(channel_indices)  
    )

    val_ds = SilentWearDataset(
        X[val_idx], y[val_idx],
        window_ms=window_ms,
        apply_preprocessing=False,
        normalize=True,
        num_channels=len(channel_indices)
    )

    return train_ds, val_ds


def plot_confusion_matrix(cm: np.ndarray, class_names: list, out_path: str):
    cm_display = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    fmt = '.2f'
    vmax = 1.0
    title = 'Confusion Matrix (normalized)'

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=vmax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close() 