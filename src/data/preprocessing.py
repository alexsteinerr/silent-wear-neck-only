"""
Ingestion pipeline for the *real* SilentWear dataset layout, as documented at
https://huggingface.co/datasets/PulpBio/SilentWear and
https://github.com/pulp-bio/SilentWear.

Expected directory layout (this is what the screenshot in the conversation
shows, and matches the official dataset card):

    <data_root>/
      S01/
        silent/
          sess_1_batch_1.h5
          sess_1_batch_2.h5
          ...
          sess_3_batch_5.h5
        vocalized/
          sess_1_batch_1.h5
          ...
        summary_file.csv
      S02/
        ...
      S03/
      S04/

Each .h5 file holds one *continuous* recording of one (session, batch) --
NOT pre-cut windows. It's read with:

    pd.read_hdf(path, key="emg")

and contains (per the dataset card):
    Ch_0..Ch_15            raw EMG samples (16 channels)
    Ch_0_filt..Ch_15_filt  filtered EMG (20 Hz high-pass + 50 Hz notch)
    Label_int, Label_str   per-sample class label (0=rest, 1..8=commands)
    session_id, batch_id   metadata

Because each file is continuous, we have to *segment* it ourselves: find
contiguous runs of the same Label_int (exactly what the paper describes as
"segmented ... starting at trigger onset and ending at the subsequent
trigger"), turn each run into one (channels, T) example, and crop/pad it to
the target window length.

IMPORTANT ASSUMPTION -- please verify:
The paper's model uses 14 effective EMG channels, but the raw files store
16 (`Ch_0`..`Ch_15`). Which 2 channels are unused/reference is not stated on
the dataset card. This script defaults to the first 14 filtered channels
(`Ch_0_filt`..`Ch_13_filt`). If your copy of the official repo's
`config/paper_models_config.yaml` specifies a different channel subset,
pass it explicitly via `channel_indices=[...]`.
"""

import argparse
import os
import re
from collections import namedtuple

import numpy as np
import pandas as pd

FILENAME_RE = re.compile(r"sess_(\d+)_batch_(\d+)\.h5$")
DEFAULT_SUBJECTS = ["S01", "S02", "S03", "S04"]
DEFAULT_CONDITIONS = ["silent", "vocalized"]
SAMPLING_RATE_HZ = 500
NUM_CHANNELS = 14  # matches SpeechNet's expected input

FileRef = namedtuple("FileRef", ["subject", "condition", "session", "batch", "path"])


def discover_h5_files(data_root, subjects=None, conditions=None):
    """Walk <data_root>/<subject>/<condition>/sess_S_batch_B.h5 and return a
    list of FileRef, sorted by (subject, condition, session, batch)."""
    subjects = subjects or DEFAULT_SUBJECTS
    conditions = conditions or DEFAULT_CONDITIONS

    refs = []
    for subject in subjects:
        for condition in conditions:
            cond_dir = os.path.join(data_root, subject, condition)
            if not os.path.isdir(cond_dir):
                print(f"warning: missing directory {cond_dir}, skipping")
                continue
            for fname in sorted(os.listdir(cond_dir)):
                m = FILENAME_RE.search(fname)
                if not m:
                    continue
                session, batch = int(m.group(1)), int(m.group(2))
                refs.append(FileRef(subject, condition, session, batch,
                                     os.path.join(cond_dir, fname)))

    refs.sort(key=lambda r: (r.subject, r.condition, r.session, r.batch))
    return refs


def _select_channel_columns(df, use_filtered=True, channel_indices=None):
    """Pick the EMG channel columns for one recording.

    If channel_indices is None, auto-detect every "Ch_{i}[_filt]" column
    actually present in the file and use all of them, sorted numerically.
    This matters because the real files do NOT contain a contiguous
    Ch_0..Ch_13 range -- e.g. Ch_11 and Ch_12 are absent (likely the
    ground-reference channels were already dropped upstream), so the 14
    real EMG channels are indices [0,1,2,3,4,5,6,7,8,9,10,13,14,15].
    Hardcoding range(14) breaks on this. Auto-detection is robust to
    whichever 14 (or other count) indices a given file actually has.
    """
    suffix = "_filt" if use_filtered else ""
    pattern = re.compile(rf"^Ch_(\d+){re.escape(suffix)}$")

    if channel_indices is None:
        available = []
        for col in df.columns:
            m = pattern.match(col)
            if m:
                available.append(int(m.group(1)))
        available.sort()
        if not available:
            raise ValueError(
                f"no columns matching 'Ch_<i>{suffix}' found. "
                f"Available columns: {list(df.columns)}"
            )
        if len(available) != NUM_CHANNELS:
            print(f"warning: found {len(available)} channels {available}, "
                  f"expected {NUM_CHANNELS} (model input channels). "
                  f"Proceeding with all {len(available)} -- update SpeechNet's "
                  f"NUM_CHANNELS/phi4-phi5 if this count differs across your files.")
        channel_indices = available

    cols = [f"Ch_{i}{suffix}" for i in channel_indices]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"columns not found in file: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return cols


def load_h5_recording(path, use_filtered=True, channel_indices=None):
    """Read one continuous recording. Returns:
        emg: (C, N_samples) float32
        label_int: (N_samples,) int64
        label_str: (N_samples,) object (or None if column absent)
    """
    df = pd.read_hdf(path, key="emg")
    cols = _select_channel_columns(df, use_filtered=use_filtered, channel_indices=channel_indices)

    emg = df[cols].to_numpy(dtype=np.float32).T  # (C, N)
    label_int = df["Label_int"].to_numpy(dtype=np.int64)
    label_str = df["Label_str"].to_numpy() if "Label_str" in df.columns else None
    return emg, label_int, label_str


def segment_by_label(emg, label_int):
    """Split a continuous (C, N) recording into contiguous same-label runs.

    Returns a list of (start_idx, end_idx, label) with end_idx exclusive --
    this mirrors the paper's "windows starting at trigger onset and ending
    at the subsequent trigger" segmentation.
    """
    if len(label_int) == 0:
        return []

    change_points = np.where(np.diff(label_int) != 0)[0] + 1
    boundaries = np.concatenate(([0], change_points, [len(label_int)]))

    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segments.append((int(start), int(end), int(label_int[start])))
    return segments


def build_dataset(data_root, subjects=None, conditions=None, window_ms=1400,
                   use_filtered=True, channel_indices=None,
                   rest_label=0, rest_keep_fraction=1.0, seed=0,
                   min_segment_samples=10):
    """Walk the dataset, segment every recording, and return fixed-length
    windows ready for SilentWearDataset / SpeechNet.

    Returns
    -------
    X : np.ndarray, shape (N, C, T)
    y : np.ndarray, shape (N,)
    meta : dict of np.ndarray, each shape (N,):
        'subject', 'condition', 'session', 'batch'
        -- use these to build Global / Inter-Session splits faithfully,
        e.g. held-out session for leave-one-session-out.
    """
    rng = np.random.RandomState(seed)
    refs = discover_h5_files(data_root, subjects, conditions)
    if not refs:
        raise FileNotFoundError(
            f"no sess_*_batch_*.h5 files found under {data_root} "
            f"(checked subjects={subjects or DEFAULT_SUBJECTS}, "
            f"conditions={conditions or DEFAULT_CONDITIONS})"
        )

    T = int(round(window_ms * SAMPLING_RATE_HZ / 1000.0))

    # Lock channel_indices from the first file and reuse for every
    # subsequent file, so all recordings yield the same channel count/order
    # even if individual files were saved with slightly different columns.
    if channel_indices is None:
        first_df = pd.read_hdf(refs[0].path, key="emg")
        cols = _select_channel_columns(first_df, use_filtered=use_filtered, channel_indices=None)
        suffix = "_filt" if use_filtered else ""
        channel_indices = [int(c[len("Ch_"):-len(suffix)] if suffix else c[len("Ch_"):]) for c in cols]
        print(f"auto-detected channel indices (locked for all files): {channel_indices}")

    X_list, y_list = [], []
    subj_list, cond_list, sess_list, batch_list = [], [], [], []

    for ref in refs:
        emg, label_int, _ = load_h5_recording(
            ref.path, use_filtered=use_filtered, channel_indices=channel_indices)

        for start, end, label in segment_by_label(emg, label_int):
            if end - start < min_segment_samples:
                continue  # skip degenerate/edge segments

            if label == rest_label and rest_keep_fraction < 1.0:
                if rng.rand() > rest_keep_fraction:
                    continue

            window = emg[:, start:end]
            window = _fit_length(window, T)

            X_list.append(window)
            y_list.append(label)
            subj_list.append(ref.subject)
            cond_list.append(ref.condition)
            sess_list.append(ref.session)
            batch_list.append(ref.batch)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    meta = {
        "subject": np.array(subj_list),
        "condition": np.array(cond_list),
        "session": np.array(sess_list, dtype=np.int64),
        "batch": np.array(batch_list, dtype=np.int64),
    }
    return X, y, meta


def _fit_length(window, T):
    C, n = window.shape
    if n == T:
        return window
    if n > T:
        start = (n - T) // 2
        return window[:, start:start + T]
    pad_total = T - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(window, ((0, 0), (pad_left, pad_right)), mode="constant")


def main():
    p = argparse.ArgumentParser(description="Build .npy training data from SilentWear .h5 recordings")
    p.add_argument("--data_root", type=str, required=True,
                    help="path to data_raw_and_filt (contains S01/, S02/, ...)")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--subjects", type=str, default=None, help="comma-separated, e.g. S01,S02")
    p.add_argument("--conditions", type=str, default="silent,vocalized")
    p.add_argument("--window_ms", type=int, default=1400)
    p.add_argument("--use_raw", action="store_true", help="use raw channels instead of filtered")
    p.add_argument("--channel_indices", type=str, default=None,
                    help="comma-separated channel indices, e.g. 0,1,2,...,13 (default: first 14)")
    p.add_argument("--rest_keep_fraction", type=float, default=1.0,
                    help="randomly subsample the rest class to this fraction (paper balances rest vs commands)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    subjects = args.subjects.split(",") if args.subjects else None
    conditions = args.conditions.split(",")
    channel_indices = ([int(c) for c in args.channel_indices.split(",")]
                        if args.channel_indices else None)

    os.makedirs(args.out_dir, exist_ok=True)

    for condition in conditions:
        print(f"processing condition={condition} ...")
        X, y, meta = build_dataset(
            args.data_root, subjects=subjects, conditions=[condition],
            window_ms=args.window_ms, use_filtered=not args.use_raw,
            channel_indices=channel_indices,
            rest_keep_fraction=args.rest_keep_fraction, seed=args.seed,
        )
        print(f"  {condition}: X={X.shape}, classes={sorted(set(y.tolist()))}")

        np.save(os.path.join(args.out_dir, f"X_{condition}.npy"), X)
        np.save(os.path.join(args.out_dir, f"y_{condition}.npy"), y)
        np.savez(os.path.join(args.out_dir, f"meta_{condition}.npz"), **meta)

    print(f"done. Saved to {args.out_dir}")


if __name__ == "__main__":
    main()