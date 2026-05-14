import os
import torch

# Paths anchored to this file so the activity works from any cwd.
ACTIVITY_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR  = os.path.join(ACTIVITY_DIR, "dataset")
MODELS_DIR   = os.path.join(ACTIVITY_DIR, "checkpoints")
RESULTS_DIR  = os.path.join(ACTIVITY_DIR, "results")

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Tickers ---
# TSLA and DOGE-USD are the targets (Musk talks about them).
# AAPL is the negative control (Musk does not tweet about Apple).
TICKERS = ["TSLA", "DOGE-USD", "AAPL"]
START_DATE = "2017-01-01"
END_DATE   = "2025-04-30"

# --- Series features (price/volume only) ---
WINDOW = 10
PRICE_FEATURES = [
    "log_return",
    "intraday_range",
    "vol_5d",
    "vol_20d",
    "volume_ratio",
    "rel_open",
    "rel_high",
    "rel_low",
]
SENTIMENT_FEATURES = ["sent_pos", "sent_neg", "sent_neu"]
EMOTION_FEATURES   = ["emo_joy", "emo_optimism", "emo_anger", "emo_sadness"]
TOXICITY_FEATURES  = ["tox_toxic", "tox_severe_toxic", "tox_obscene",
                      "tox_threat", "tox_insult", "tox_identity_hate"]
ALL_TEXT_FEATURES  = SENTIMENT_FEATURES + EMOTION_FEATURES + TOXICITY_FEATURES

# --- Label: shock detector on next-day realized volatility ---
LABEL_ROLLING_WINDOW = 60
SHOCK_PERCENTILE     = 0.80
SHOCK_POS_WEIGHT     = 4.0

# --- Temporal splits (no leakage). Bounded by tweet coverage. ---
TRAIN_END = "2023-12-31"
VAL_END   = "2024-08-31"

# --- Text / FinBERT (inference only, not fine-tuned) ---
TEXT_BACKBONE_NAME  = "ProsusAI/finbert"
EMOTION_MODEL_NAME  = "cardiffnlp/twitter-roberta-base-emotion"
TOXICITY_MODEL_NAME = "unitary/toxic-bert"
MAX_TEXT_TOKENS     = 128
SENTIMENT_BATCH     = 32
TWEETS_HF_REPO = "fdaudens/musk-tweets"
TWEETS_HF_FILE = "elonmusk_data_fixed - Sheet1.csv"

# --- Part 1: classical baselines ---
TFIDF_MAX_FEATURES = 1000
LOGREG_C           = 1.0
LOGREG_MAX_ITER    = 1000

# --- Parts 2 and 3: LSTM ---
LSTM_HIDDEN_DIM    = 128
LSTM_NUM_LAYERS    = 2
LSTM_DROPOUT       = 0.3
LSTM_BIDIRECTIONAL = False
LSTM_EPOCHS        = 40
LSTM_BATCH         = 64
LSTM_LR            = 1e-3
LSTM_WD            = 1e-4

# --- Training common ---
EARLY_STOP_PATIENCE = 6


def get_device() -> torch.device:
    """
    Pick the best available backend:
      1. CUDA (NVIDIA)
      2. MPS  (Apple Silicon, Metal)
      3. CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
