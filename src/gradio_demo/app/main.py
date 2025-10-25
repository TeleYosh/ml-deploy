import gradio as gr
import numpy as np
import torch 
import torch.nn.functional as F
import torchvision.transforms as T
import json
import os
from pathlib import Path
from datasets import load_dataset
from gradio_demo.model.sketch_images.cnn import CNN
from PIL import Image


# Load your model
n_classes = 345
params = {
    'n_filters': 30,
    'hidden_dim': 100,
    'n_layers': 2,
    'n_classes': n_classes
}
model = CNN(**params)
ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "gradio_demo" / "model" / "sketch_images" / "weights" / "cnn.pth"
model.load_state_dict(torch.load(PATH, map_location='cpu'))
model.eval()

# utils
# Get the directory of the current script
current_dir = Path(__file__).parent
labels_path = current_dir / '../data/labels.json'
with open(labels_path, 'r') as f:
    names = json.load(f)

transform = T.Compose([
    T.ToTensor(),                            # (1, H, W), values in [0, 1], white=1 black=0
    T.Lambda(lambda x: 1.0 - x),             # invert -> white=0, black=1 
    T.Resize((28, 28), interpolation=T.InterpolationMode.BILINEAR),
    # T.Normalize((0.5,), (0.5,))            # optional if your model expects [-1, 1]
])

# some examples
# examples_images
folder_path = current_dir / '../data/examples_images/'
file_names = os.listdir(folder_path)
# print(f'file names {file_names}')
example_images = [np.array(Image.open(folder_path/image_file)) for image_file in file_names]

def predict(input_image):
    img = input_image['composite']
    if img is None:
        return {"No drawing detected": 1.0}
    img = transform(img)
    img = img.unsqueeze(0).to(torch.float32) # add batch dimension
    # torch.save(img, )
    with torch.no_grad():
        out = model(img)
    # idx = torch.argmax(out).item()
    probs = F.softmax(out, dim=1).squeeze(0)
    res = {names[i]:proba.item() for i, proba in enumerate(probs)}
    return res

demo = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(
        label="Draw a sketch",
        image_mode='L',
        brush=gr.Brush(default_size=20, default_color='black', colors=['black'], color_mode='fixed')
        ),
    outputs=gr.Label(num_top_classes=5),
    title="Sketch Recognition model",
    examples=example_images,
    clear_btn=gr.ClearButton(),
    live=True
)

if __name__ == "__main__":
    demo.launch()
