# Baseline Late Fusion configuration

class UNetConfig:
    INPUT_CHANNELS = 1              # single modality per expert
    N_CLASSES = 3                   # WT, TC, ET
    N_STAGES = 6
    N_FEATURES_PER_STAGE = [8, 16, 32, 64, 80, 80]  # lightweight
    KERNEL_SIZES = [[3, 3, 3]] * 6
    STRIDES = [[1, 1, 1], *[[2, 2, 2]] * 5]
    APPLY_DEEP_SUPERVISION = False


class DatasetConfig:
    DATASET_DIR = 'assets/data'     # symlink to nnUNet_preprocessed_BraTS2018
    SPLITS_FILE_PATH = 'assets/kfold_splits.json'
    DROP_MODE = None                # use all 4 modalities
    POSSIBLE_DROPPED_MODALITY_COMBINATIONS = [
        [], [0], [1], [2], [3],
        [0, 1], [0, 2], [0, 3],
        [1, 2], [1, 3], [2, 3],
        [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]
    ]
    FOLD = 2    # cross-validation fold 2


class TrainingConfig:
    RANDOM_SEED = 12345
    N_EPOCHS = 500
    LEARNING_RATE = 0.001
    BATCH_SIZE = 4
    VAL_BATCH_SIZE = 8
    VAL_FREQ = 5
    RESULTS_DIR = 'saved_models/baseline'
    APPLY_EARLY_STOPPING = False
