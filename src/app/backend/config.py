from pathlib import Path

# Locate the root of the entire repo
BACKEND_DIR = Path(__file__).resolve().parent          # app/backend
APP_DIR = BACKEND_DIR.parent                           # app/
SRC_DIR = APP_DIR.parent  
ROOT = SRC_DIR.parent                                 # ← repo root

# Model + data paths
MODEL_DIR =  SRC_DIR / "models" / "weights"
MODEL_PATH = MODEL_DIR / "cnn_45_acc.pth"

PARAMS_PATH =  SRC_DIR / "models" / 'params' / 'cnn_45_params.json'

DATA_DIR = ROOT / "data"
LABELS_PATH = DATA_DIR / "labels.json"
