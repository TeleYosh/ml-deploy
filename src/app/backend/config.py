from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "model" / "sketch_images" / "weights" / "cnn.pth"
LABELS_PATH = ROOT / "data" / "labels.json"
