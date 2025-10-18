import gradio as gr
import numpy as np
import torch 
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

def predict(input_image):
    img = input_image['composite']
    # print(f'img {img.shape} type {type(img)}')
    img = img / 255.0
    img = np.resize(img, (28, 28))
    img = torch.from_numpy(img)
    # print(f'type {type(img)} shape {img.shape}')
    img = img.unsqueeze(0).unsqueeze(0).to(torch.float32)
    # print('-'*50)
    # print(f'type {type(img)} shape {img.shape} dtype {img.dtype}')
    # print('-'*50)
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
