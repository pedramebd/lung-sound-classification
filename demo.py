"""
demo.py — End-to-end respiratory sound classification demo.

Loads a trained model and classifies respiratory cycles from an audio file.

Usage:
    python demo.py --audio_path path/to/audio.wav
    python demo.py --audio_path path/to/audio.wav --annotation_path path/to/annotation.txt
    python demo.py --example  (runs on a sample from the dataset)

Requirements:
    - Trained models in outputs/models/ (run notebooks 01-07 first)
    - Dataset accessible at the path specified in src/config.py
"""

import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import joblib
from pathlib import Path

from src.config import (
    AUDIO_DIR, CYCLE_CLASS_NAMES, FIGURES_DIR, HOP_LENGTH,
    MODELS_DIR, N_MELS, N_FFT, SAMPLE_RATE,
)
from src.data_loader import parse_annotation_file, build_cycle_dataset
from src.feature_extraction import extract_handcrafted_features


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

CYCLE_DURATION = 4.0
TARGET_LEN = int(CYCLE_DURATION * SAMPLE_RATE)

COLOURS = {
    'Normal': '#2ecc71',
    'Crackle': '#e74c3c',
    'Wheeze': '#3498db',
    'Both': '#9b59b6',
}


# ──────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────

def load_models():
    """Load trained SVM model and scaler."""
    scaler_path = MODELS_DIR / 'scaler.joblib'
    model_path = MODELS_DIR / 'best_ml_model.joblib'

    if not scaler_path.exists() or not model_path.exists():
        print("ERROR: Trained models not found in outputs/models/")
        print("Please run the training notebooks (01-07) first.")
        sys.exit(1)

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    print("Loaded SVM model and feature scaler.")
    return model, scaler


def segment_audio(audio_path, annotation_path=None):
    """
    Segment audio into respiratory cycles.

    If annotation_path is provided, use expert annotations.
    Otherwise, segment into fixed-length windows.
    """
    y_full, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    duration = len(y_full) / sr

    segments = []

    if annotation_path and Path(annotation_path).exists():
        # Use expert annotations
        annotations = parse_annotation_file(Path(annotation_path))
        print(f"Using {len(annotations)} annotated respiratory cycles.")

        for _, row in annotations.iterrows():
            start, end = row['start'], row['end']
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            y_segment = y_full[start_sample:end_sample]

            # Pad or truncate
            if len(y_segment) < TARGET_LEN:
                y_segment = np.pad(y_segment, (0, TARGET_LEN - len(y_segment)))
            else:
                y_segment = y_segment[:TARGET_LEN]

            segments.append({
                'audio': y_segment,
                'start': start,
                'end': end,
                'ground_truth': row.get('label', None),
            })
    else:
        # Fixed-length windowing (3-second windows with 1s overlap)
        window_sec = 3.0
        hop_sec = 2.0
        print(f"No annotations provided. Using {window_sec}s sliding windows.")

        start = 0
        while start + window_sec <= duration:
            start_sample = int(start * sr)
            end_sample = int((start + window_sec) * sr)
            y_segment = y_full[start_sample:end_sample]

            if len(y_segment) < TARGET_LEN:
                y_segment = np.pad(y_segment, (0, TARGET_LEN - len(y_segment)))
            else:
                y_segment = y_segment[:TARGET_LEN]

            segments.append({
                'audio': y_segment,
                'start': start,
                'end': start + window_sec,
                'ground_truth': None,
            })
            start += hop_sec

    print(f"Total segments: {len(segments)}")
    return segments, y_full


def classify_segments(segments, model, scaler):
    """Extract features and classify each segment."""
    results = []

    for i, seg in enumerate(segments):
        # Extract handcrafted features
        features = extract_handcrafted_features(seg['audio'], sr=SAMPLE_RATE)
        features_scaled = scaler.transform(features.reshape(1, -1))

        # Predict
        prediction = model.predict(features_scaled)[0]
        class_name = CYCLE_CLASS_NAMES[prediction]

        # Binary
        is_abnormal = prediction > 0

        results.append({
            'segment_idx': i,
            'start': seg['start'],
            'end': seg['end'],
            'prediction': prediction,
            'class_name': class_name,
            'is_abnormal': is_abnormal,
            'ground_truth': seg['ground_truth'],
        })

    return results


def visualise_results(y_full, results, audio_path, save_path=None):
    """Visualise the classification results."""
    duration = len(y_full) / SAMPLE_RATE

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1, 1]})

    # Waveform with colour-coded segments
    time = np.arange(len(y_full)) / SAMPLE_RATE
    axes[0].plot(time, y_full, color='gray', alpha=0.5, linewidth=0.5)

    for res in results:
        start_sample = int(res['start'] * SAMPLE_RATE)
        end_sample = min(int(res['end'] * SAMPLE_RATE), len(y_full))
        colour = COLOURS[res['class_name']]
        axes[0].axvspan(res['start'], res['end'], alpha=0.25, color=colour)

    axes[0].set_xlim(0, duration)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Respiratory Sound Classification — {Path(audio_path).stem}')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLOURS[c], alpha=0.4, label=c) for c in CYCLE_CLASS_NAMES]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y_full, sr=SAMPLE_RATE, n_mels=N_MELS,
                                         n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                             x_axis='time', y_axis='mel', ax=axes[1])
    axes[1].set_title('Mel Spectrogram')

    # Classification timeline
    for res in results:
        colour = COLOURS[res['class_name']]
        axes[2].barh(0, res['end'] - res['start'], left=res['start'],
                     color=colour, edgecolor='white', linewidth=0.5, height=0.6)
        # Label
        mid = (res['start'] + res['end']) / 2
        if res['end'] - res['start'] > 0.5:
            axes[2].text(mid, 0, res['class_name'][0], ha='center', va='center',
                         fontsize=8, fontweight='bold', color='white')

    axes[2].set_xlim(0, duration)
    axes[2].set_ylim(-0.5, 0.5)
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_title('Cycle-by-Cycle Classification')
    axes[2].set_yticks([])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualisation: {save_path}")

    plt.show()
    return fig


def print_summary(results):
    """Print classification summary."""
    total = len(results)
    counts = {}
    for r in results:
        counts[r['class_name']] = counts.get(r['class_name'], 0) + 1

    abnormal = sum(1 for r in results if r['is_abnormal'])
    normal = total - abnormal

    print("\n" + "=" * 50)
    print("  CLASSIFICATION SUMMARY")
    print("=" * 50)
    print(f"  Total cycles analysed: {total}")
    print(f"  Normal: {normal} ({normal/total*100:.1f}%)")
    print(f"  Abnormal: {abnormal} ({abnormal/total*100:.1f}%)")
    print()

    for cls in CYCLE_CLASS_NAMES:
        count = counts.get(cls, 0)
        bar = '█' * int(count / total * 30) if total > 0 else ''
        print(f"  {cls:<10} {count:>4} ({count/total*100:>5.1f}%)  {bar}")

    # Check ground truth if available
    gt_available = [r for r in results if r['ground_truth'] is not None]
    if gt_available:
        correct = sum(1 for r in gt_available if r['prediction'] == r['ground_truth'])
        print(f"\n  Ground truth available: {len(gt_available)} cycles")
        print(f"  Correct predictions: {correct}/{len(gt_available)} ({correct/len(gt_available)*100:.1f}%)")

    # Clinical recommendation
    print("\n" + "-" * 50)
    if abnormal / total > 0.3:
        print("  ⚠ Significant abnormal sounds detected.")
        print("  Recommendation: Further clinical evaluation advised.")
    elif abnormal / total > 0.1:
        print("  ⚡ Some abnormal sounds detected.")
        print("  Recommendation: Monitor and reassess if symptoms persist.")
    else:
        print("  ✓ Predominantly normal respiratory sounds.")
        print("  Recommendation: No immediate concern from audio analysis.")

    print("\n  Note: This is a research tool, not a medical diagnostic device.")
    print("=" * 50)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Respiratory Sound Classification Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py --audio_path recording.wav
  python demo.py --audio_path recording.wav --annotation_path recording.txt
  python demo.py --example
        """
    )
    parser.add_argument('--audio_path', type=str, help='Path to .wav audio file')
    parser.add_argument('--annotation_path', type=str, default=None,
                        help='Path to annotation .txt file (optional)')
    parser.add_argument('--example', action='store_true',
                        help='Run on a sample from the ICBHI dataset')
    parser.add_argument('--save', action='store_true',
                        help='Save visualisation to outputs/figures/')

    args = parser.parse_args()

    print("=" * 50)
    print("  Lung Sound Classification Demo")
    print("  ICBHI 2017 | Patient-Aware Evaluation")
    print("=" * 50)

    # Load model
    model, scaler = load_models()

    # Get audio
    if args.example:
        # Use a sample from the dataset
        cycle_df = build_cycle_dataset()
        sample = cycle_df.iloc[0]
        audio_path = sample['audio_path']
        annotation_path = str(Path(audio_path).with_suffix('.txt'))
        print(f"\nUsing example: {Path(audio_path).stem}")
        print(f"Patient ID: {sample['patient_id']}")
    elif args.audio_path:
        audio_path = args.audio_path
        annotation_path = args.annotation_path
    else:
        parser.print_help()
        sys.exit(1)

    # Process
    print(f"\nProcessing: {audio_path}")
    segments, y_full = segment_audio(audio_path, annotation_path)
    results = classify_segments(segments, model, scaler)

    # Output
    print_summary(results)

    save_path = None
    if args.save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / f"demo_{Path(audio_path).stem}.png"

    visualise_results(y_full, results, audio_path, save_path)


if __name__ == '__main__':
    main()
