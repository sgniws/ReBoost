"""Boosted Late Fusion (Sustained Boosting + ACA) configuration."""
from configs import UNetConfig, DatasetConfig


class BoostedConfig:
    """Sustained Boosting hyperparameters."""

    # Sustained Boosting
    LAMBDA_SMOOTH = 0.33
    HEAD_HIDDEN_CHANNELS = 8
    MAX_HEADS_PER_MODALITY = 10

    # ACA
    ACA_SIGMA = 1.0
    ACA_TAU = 0.01
    ACA_CHECK_INTERVAL = 30

    # Training (aligned with baseline)
    RANDOM_SEED = 12345
    N_EPOCHS = 300
    LEARNING_RATE = 0.001
    BATCH_SIZE = 4
    VAL_BATCH_SIZE = 8
    VAL_FREQ = 5
    RESULTS_DIR = 'saved_models/boosted'
    APPLY_EARLY_STOPPING = False
