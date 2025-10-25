import gradio as gr
import numpy as np
import torch 
import torch.nn.functional as F
import torchvision.transforms as T
from pathlib import Path
from datasets import load_dataset
from gradio_demo.model.sketch_images.cnn import CNN


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
dataset = load_dataset("Xenova/quickdraw-small")
train_dataset = dataset['train']
names = train_dataset.features['label'].names
transform = T.Compose([
    T.ToTensor(),                            # (1, H, W), values in [0, 1], white=1 black=0
    # T.Lambda(lambda x: x/255.0), 
    T.Lambda(lambda x: 1.0 - x),             # invert -> white=0, black=1 ✅
    T.Resize((28, 28), interpolation=T.InterpolationMode.BILINEAR),
    # T.Normalize((0.5,), (0.5,))            # optional if your model expects [-1, 1]
])

# img_tensor = transform(img)

def predict(input_image):
    img = input_image['composite']
    # print(f'img {img.shape} type {type(img)}')
    # img = img / 255.0
    # img = torch.from_numpy(img)
    # print(f'OG image model type {img}')
    # img = img.unsqueeze(0)
    # resize = T.Resize((28, 28), interpolation=T.InterpolationMode.BICUBIC, antialias=True)
    # img = resize(img)
    img = transform(img)
    print('-'*50)
    print('-'*50)
    print(f'type {type(img)} shape {img.shape}')
    print('-'*50)
    print('-'*50)
    print('-'*50)
    img = img.unsqueeze(0).to(torch.float32)
    print('-'*50)
    print(f'before model type {type(img)} shape {img.shape} dtype {img.dtype}')
    print('-'*50)
    print('-'*50)
    print(f'image model type {img}')
    print('-'*50)
    with torch.no_grad():
        out = model(img)
    idx = torch.argmax(out).item()
    return names[idx]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(label="Draw a Digit a khawa diali", image_mode='L'),
    outputs=gr.Label(num_top_classes=1),
    title="Digit Classifier"
)

if __name__ == "__main__":
    demo.launch()
