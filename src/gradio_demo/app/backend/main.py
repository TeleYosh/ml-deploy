from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pathlib import Path
from gradio_demo.model.sketch_images.cnn import CNN

import torch
import torchvision.transforms as T
import torch.nn.functional as F
import json
import io
import torchvision.utils as vutils

app = FastAPI()

# allow requests from frontend - Need to understand it later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load torch model
n_classes = 345
params = {
    'n_filters': 30,
    'hidden_dim': 100,
    'n_layers': 2,
    'n_classes': n_classes
}
model = CNN(**params)
ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "model" / "sketch_images" / "weights" / "cnn.pth"
model.load_state_dict(torch.load(PATH, map_location='cpu'))
model.eval()

# utils
current_dir = Path(__file__).parent
labels_path = current_dir / '../../data/labels.json'
with open(labels_path, 'r') as f:
    names = json.load(f)

transform = T.Compose([
    T.ToTensor(),                            # (1, H, W), values in [0, 1], white=1 black=0
    T.Lambda(lambda x: 1.0 - x),             # invert -> white=0, black=1 
    T.Resize((28, 28), interpolation=T.InterpolationMode.BILINEAR),
    # T.Normalize((0.5,), (0.5,))            # optional if your model expects [-1, 1]
])

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('L')
    img = transform(image).unsqueeze(0).to(torch.float32)

    with torch.no_grad():
        out = model(img)
    idx = torch.argmax(out).item()
    probs = F.softmax(out, dim=1).squeeze(0)
    res = {names[i]:proba.item() for i, proba in enumerate(probs)}
    top5_names = sorted(res, key=lambda x:res[x], reverse=True)[:5]
    top5_probas = [res[name] for name in  top5_names]
    # return {
    #     'prediction': names[idx],
    #     'proba': probs[idx].item()
    #     }
    return {
        'predictions': top5_names,
        'probas': top5_probas
    }