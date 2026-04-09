"""
Data loading and parsing utilities for the ICBHI Respiratory Sound Database.

Handles:
  - Parsing annotation .txt files (cycle boundaries + crackle/wheeze labels)
  - Loading patient diagnosis and demographic information
  - Segmenting full recordings into individual respiratory cycles
  - Patient-aware train/val/test splitting (no data leakage)
"""

import re
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    AUDIO_DIR,
    CYCLE_CLASSES,
    CYCLE_DURATION,
    DEMOGRAPHICS_FILE,
    DIAGNOSIS_FILE,
    RANDOM_SEED,
    SAMPLE_RATE,
    TEST_SIZE,
    VAL_SIZE,
)


# ──────────────────────────────────────────────
# Filename parsing
# ──────────────────────────────────────────────
# Format: PatientID_RecordingIndex_ChestLocation_AcquisitionMode_RecordingEquipment
FILENAME_PATTERN = re.compile(
    r"^(\d+)_(\w+)_(\w+)_(\w+)_(\w+)$"
)


def parse_filename(stem: str) -> dict:
    """Extract metadata from a recording filename (without extension)."""
    match = FILENAME_PATTERN.match(stem)
    if not match:
        return {"patient_id": int(stem.split("_")[0])}
    return {
        "patient_id": int(match.group(1)),
        "recording_index": match.group(2),
        "chest_location": match.group(3),
        "acquisition_mode": match.group(4),
        "recording_equipment": match.group(5),
    }


# ──────────────────────────────────────────────
# Annotation parsing
# ──────────────────────────────────────────────

def parse_annotation_file(txt_path: Path) -> pd.DataFrame:
    """
    Parse a single annotation .txt file.

    Each line: start_time  end_time  crackle(0/1)  wheeze(0/1)

    Returns DataFrame with columns:
        start, end, crackle, wheeze, label (int class index)
    """
    rows = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            start, end = float(parts[0]), float(parts[1])
            crackle, wheeze = int(parts[2]), int(parts[3])
            label = CYCLE_CLASSES[(crackle, wheeze)]
            rows.append({
                "start": start,
                "end": end,
                "crackle": crackle,
                "wheeze": wheeze,
                "label": label,
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Diagnosis & demographics
# ──────────────────────────────────────────────

def load_diagnosis(path: Optional[Path] = None) -> pd.DataFrame:
    """Load patient_diagnosis.csv → DataFrame with patient_id and diagnosis."""
    path = path or DIAGNOSIS_FILE
    df = pd.read_csv(
        path, names=["patient_id", "diagnosis"], sep=",", header=None
    )
    df["patient_id"] = df["patient_id"].astype(int)
    return df


def load_demographics(path: Optional[Path] = None) -> pd.DataFrame:
    """Load demographic_info.txt → DataFrame."""
    path = path or DEMOGRAPHICS_FILE
    df = pd.read_csv(
        path,
        names=["patient_id", "age", "sex", "adult_bmi", "child_weight", "child_height"],
        sep=r"\s+",
        header=None,
    )
    df["patient_id"] = df["patient_id"].astype(int)
    return df


# ──────────────────────────────────────────────
# Build master cycle-level dataset
# ──────────────────────────────────────────────

def build_cycle_dataset(audio_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Iterate over all annotation files and build a DataFrame where each row
    represents one respiratory cycle with metadata.

    Columns:
        audio_path, annotation_path, patient_id, chest_location,
        acquisition_mode, recording_equipment,
        start, end, duration, crackle, wheeze, label
    """
    audio_dir = audio_dir or AUDIO_DIR
    records = []

    txt_files = sorted(audio_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} annotation files.")

    for txt_path in txt_files:
        wav_path = txt_path.with_suffix(".wav")
        if not wav_path.exists():
            continue

        meta = parse_filename(txt_path.stem)
        cycles = parse_annotation_file(txt_path)

        for _, row in cycles.iterrows():
            record = {
                "audio_path": str(wav_path),
                "annotation_path": str(txt_path),
                **meta,
                "start": row["start"],
                "end": row["end"],
                "duration": row["end"] - row["start"],
                "crackle": row["crackle"],
                "wheeze": row["wheeze"],
                "label": row["label"],
            }
            records.append(record)

    df = pd.DataFrame(records)
    print(f"Total respiratory cycles: {len(df)}")
    return df


# ──────────────────────────────────────────────
# Audio segment loading
# ──────────────────────────────────────────────

def load_cycle_audio(
    audio_path: str,
    start: float,
    end: float,
    sr: int = SAMPLE_RATE,
    target_duration: float = CYCLE_DURATION,
) -> np.ndarray:
    """
    Load a single respiratory cycle from a recording.

    - Extracts segment [start, end]
    - Resamples to `sr`
    - Pads or truncates to `target_duration` seconds
    """
    duration = end - start
    y, _ = librosa.load(audio_path, sr=sr, offset=start, duration=duration)

    target_len = int(target_duration * sr)

    if len(y) < target_len:
        # Pad with zeros
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        # Truncate
        y = y[:target_len]

    return y


# ──────────────────────────────────────────────
# Patient-aware splitting
# ──────────────────────────────────────────────

def patient_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data ensuring no patient appears in multiple sets.

    Returns (train_df, val_df, test_df).
    """
    # First split: train+val vs test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss1.split(df, groups=df["patient_id"]))

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Second split: train vs val (relative to train+val)
    relative_val_size = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx, val_idx = next(gss2.split(train_val_df, groups=train_val_df["patient_id"]))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    print(f"Split sizes — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Unique patients — Train: {train_df['patient_id'].nunique()}, "
          f"Val: {val_df['patient_id'].nunique()}, Test: {test_df['patient_id'].nunique()}")

    return train_df, val_df, test_df
