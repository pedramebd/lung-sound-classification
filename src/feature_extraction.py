"""
Feature extraction for respiratory sound classification.

Two pipelines:
  1. Handcrafted features → flat vector per cycle (for ML models)
  2. Mel spectrogram → 2D image per cycle (for CNN models)
"""

import librosa
import numpy as np

from src.config import HOP_LENGTH, N_FFT, N_MELS, N_MFCC, SAMPLE_RATE


# ──────────────────────────────────────────────
# Handcrafted feature extraction
# ──────────────────────────────────────────────

def extract_handcrafted_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract a comprehensive set of audio features from a waveform.

    Features (193-dimensional vector):
        - MFCCs (40) + delta MFCCs (40) + delta-delta MFCCs (40)  → 120
        - Chroma (12)
        - Mel spectrogram mean across time (128) → reduced to 20 via stats
        - Spectral centroid, bandwidth, contrast (7), rolloff  → 10
        - Zero-crossing rate (1)
        - RMS energy (1)
        Total: ~164 features (varies slightly with spectral contrast bands)

    All features are summarised as mean + std over time frames.
    """
    features = []

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)

    for feat in [mfccs, mfccs_delta, mfccs_delta2]:
        features.append(np.mean(feat, axis=1))
        features.append(np.std(feat, axis=1))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.append(np.mean(chroma, axis=1))
    features.append(np.std(chroma, axis=1))

    # Spectral centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.append(np.mean(cent, axis=1))
    features.append(np.std(cent, axis=1))

    # Spectral bandwidth
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.append(np.mean(bw, axis=1))
    features.append(np.std(bw, axis=1))

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.append(np.mean(contrast, axis=1))
    features.append(np.std(contrast, axis=1))

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.append(np.mean(rolloff, axis=1))
    features.append(np.std(rolloff, axis=1))

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    features.append(np.mean(zcr, axis=1))
    features.append(np.std(zcr, axis=1))

    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    features.append(np.mean(rms, axis=1))
    features.append(np.std(rms, axis=1))

    return np.concatenate(features)


# ──────────────────────────────────────────────
# Mel spectrogram extraction
# ──────────────────────────────────────────────

def extract_mel_spectrogram(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """
    Compute log-mel spectrogram for CNN input.

    Returns:
        2D array of shape (n_mels, time_steps) in dB scale.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec


def extract_mfcc_spectrogram(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """
    Compute MFCC spectrogram (alternative CNN input).

    Returns:
        2D array of shape (n_mfcc, time_steps).
    """
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    return mfccs


# ──────────────────────────────────────────────
# Batch extraction
# ──────────────────────────────────────────────

def extract_features_batch(
    audio_segments: list[np.ndarray],
    mode: str = "handcrafted",
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Extract features from a list of audio segments.

    Args:
        audio_segments: list of 1D numpy arrays (waveforms)
        mode: 'handcrafted' for ML features, 'mel_spectrogram' for CNN input
        sr: sample rate

    Returns:
        For 'handcrafted': 2D array (n_samples, n_features)
        For 'mel_spectrogram': 3D array (n_samples, n_mels, time_steps)
    """
    extractor = {
        "handcrafted": extract_handcrafted_features,
        "mel_spectrogram": extract_mel_spectrogram,
        "mfcc_spectrogram": extract_mfcc_spectrogram,
    }[mode]

    features = []
    for i, y in enumerate(audio_segments):
        if (i + 1) % 500 == 0:
            print(f"  Extracted {i + 1}/{len(audio_segments)} ...")
        try:
            feat = extractor(y, sr=sr)
            features.append(feat)
        except Exception as e:
            print(f"  Warning: Failed on segment {i}: {e}")
            # Return zeros with expected shape
            if mode == "handcrafted":
                features.append(np.zeros(features[-1].shape if features else (264,)))
            else:
                features.append(np.zeros(features[-1].shape if features else (N_MELS, 216)))

    return np.array(features)
