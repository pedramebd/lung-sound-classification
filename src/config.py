"""
Configuration for Lung Sound Classification Project.
Adjust DATA_DIR to point to your extracted Kaggle dataset.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(r"E:\Cardiff Uni\Projects\Sound-based lung disease detection\archive\Respiratory_Sound_Database") # Users should update this path file on their own devices
AUDIO_DIR = DATA_DIR / "audio_and_txt_files"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"

# Metadata files
DIAGNOSIS_FILE = DATA_DIR / "patient_diagnosis.csv"
DEMOGRAPHICS_FILE = DATA_DIR / "demographic_info.txt"

# ──────────────────────────────────────────────
# Audio processing
# ──────────────────────────────────────────────
SAMPLE_RATE = 22050          # Resample all audio to this rate
CYCLE_DURATION = 5.0         # Pad/truncate cycles to fixed length (seconds)
N_MELS = 128                 # Mel spectrogram frequency bins
N_MFCC = 40                  # Number of MFCC coefficients
HOP_LENGTH = 512
N_FFT = 2048

# ──────────────────────────────────────────────
# Labels
# ──────────────────────────────────────────────
CYCLE_CLASSES = {
    (0, 0): 0,   # Normal
    (1, 0): 1,   # Crackle
    (0, 1): 2,   # Wheeze
    (1, 1): 3,   # Both
}
CYCLE_CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15              # Of the remaining training set
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 7
DEVICE = "cuda"              # Will fallback to CPU in code if unavailable
