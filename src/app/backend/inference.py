import json
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from models.cnn import CNN, CNN_v2
from app.backend.config import MODEL_PATH, LABELS_PATH, PARAMS_PATH
import io

# ---- Load model ----
n_classes = 345
with open(PARAMS_PATH, 'r') as f:
    params = json.load(f)
model = CNN(**params, n_classes=n_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ---- Load labels ----
with open(LABELS_PATH, "r") as f:
    LABELS = json.load(f)

# ---- Transform ----
transform = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: 1.0 - x),
    T.Resize((28, 28)),
])

def predict_image(file_bytes: bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("L")
    img = transform(image).unsqueeze(0).float()

    with torch.no_grad():
        out = model(img)

    probs = F.softmax(out, dim=1).squeeze(0)
    scores = {LABELS[i]: probs[i].item() for i in range(len(LABELS))}

    top5 = sorted(scores, key=scores.get, reverse=True)[:5]
    return top5, [scores[k] for k in top5]
