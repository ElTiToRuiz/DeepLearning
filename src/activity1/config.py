import os

# All paths are anchored to this file so the activity is portable
# and works regardless of where the script is launched from.
ACTIVITY_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH   = os.path.join(ACTIVITY_DIR, "dataset", "insurance.csv")
MODELS_DIR  = os.path.join(ACTIVITY_DIR, "checkpoints")
RESULTS_DIR = os.path.join(ACTIVITY_DIR, "results")

# --- Keeping things consistent ---
RANDOM_SEED = 42

# --- Data tweaks ---
TEST_SIZE = 0.2

# --- Training basics ---
EPOCHS        = 200
LEARNING_RATE = 0.01
BATCH_SIZE    = 32

# --- Model structure defaults ---
SHALLOW_HIDDEN_SIZE = 32
DEEP_HIDDEN_SIZES   = [64, 32, 16]
DROPOUT_RATE        = 0.2

# --- Optuna search settings ---
OPTUNA_TRIALS = 20
