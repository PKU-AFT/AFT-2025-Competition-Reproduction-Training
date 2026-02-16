from pathlib import Path
import torch

# DATA_PATH = [
#     Path(r"E:\_Data\selected_features_full_sample.parquet"),
#     Path(r"E:\_Data\market_features.parquet"),
# ]

DATA_PATH = [
    Path(r"/content/gdrive/MyDrive/_Data/selected_features_top100_new.parquet"),
    Path(r"/content/gdrive/MyDrive/_Data/market_features.parquet"),
]

TARGET_COLUMN = "target"

STEP_LEN = 10
FILLNA_TYPE = "ffill+bfill"
TRAIN_DAYS = 80
VAL_DAYS = 20
TEST_DAYS = 20
TEST_BATCH_SIZE = 256
BUCKETS_PER_DAY = 55  # 每日的 seconds_in_bucket 数量，用于按时间步切分窗口

SEED = 42
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100

# Group batch size (number of daily batches per outer batch)
GROUP_BATCH_SIZE = 256

NUM_WORKERS = 0
PIN_MEMORY = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Early stopping
EARLY_STOP_PATIENCE = 10  
EARLY_STOP_MIN_DELTA = 0.0  

# OUTPUT_DIR = Path("output")

OUTPUT_DIR = Path("/content/gdrive/MyDrive/_Project/output")

MODEL_DIR = OUTPUT_DIR / "models"
PRED_DIR = OUTPUT_DIR / "preds"
LOG_DIR = OUTPUT_DIR / "logs"
