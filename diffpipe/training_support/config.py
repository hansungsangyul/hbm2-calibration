import torch

# ========================================
# 경로 설정
# ========================================
TRAIN_ROOT = 'train_data'
VAL_ROOT = 'val_data'
TEST_ROOT = 'test_data'

CHECKPOINT_DIR = 'checkpoints'
RESULTS_DIR = 'results'
LOG_DIR = 'logs'

# ========================================
# 모델 설정
# ========================================
IN_CHANNELS = 3
NUM_CLASSES = 4  # normal, crack, hole, open
OUT_CHANNELS = NUM_CLASSES  # ← 이 줄 추가


# ========================================
# 학습 설정
# ========================================
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# ========================================
# 데이터 설정
# ========================================
NUM_WORKERS = 4
PIN_MEMORY = True

IMAGE_HEIGHT = 314
IMAGE_WIDTH = 520

# ========================================
# Augmentation 설정
# ========================================
AUGMENTATION = {
    'horizontal_flip': 0.5,
    'vertical_flip': 0.5,
    'rotate90': 0.5,
    'shift_scale_rotate': 0.5,
    'noise_blur': 0.3,
    'brightness_contrast': 0.3,
}

# ========================================
# 최적화 설정
# ========================================
USE_AMP = True  # Mixed Precision Training
GRADIENT_CLIP = 1.0

# ========================================
# 기타
# ========================================
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
