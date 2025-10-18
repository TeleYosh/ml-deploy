import torch
import gradio as gr 
from model_arch import CNN
from torchvision import transforms
import numpy as np

params = {
    'n_filters': 30,
    'hidden_dim': 100,
    'n_layers': 2
}

model = CNN(**params)
model.load_state_dict(torch.load('model/mnist/mnist_model.pth'))
model.eval()


import torch
import numpy as np
import torchvision.transforms.functional as F

def preprocess_sketch_numpy(img: np.ndarray) -> torch.Tensor:
    """
    Converts a Gradio Sketchpad RGBA image (800x800x4) into a [1,1,28,28] grayscale tensor.
    No Pillow used.
    """
    # Ensure it's a float tensor
    if img.dtype != np.float32:
        img = img.astype(np.float32)

    # Normalize to [0,1]
    if img.max() > 1:
        img /= 255.0

    # If RGBA, composite over white background
    rgb = img[..., :3]
    alpha = img[..., 3:4]
    # Composite: out = rgb*alpha + white*(1-alpha)
    img = rgb * alpha + (1 - alpha)
    img_gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    # Convert to torch tensor, add channel dimension (C,H,W)
    # t = torch.from_numpy(img_gray).unsqueeze(0)  # shape (1,800,800)

    # # Resize to 28x28 using bilinear interpolation
    # t = F.resize(t, [28, 28], antialias=True)

    # # Add batch dimension: (1,1,28,28)
    # t = t.unsqueeze(0)

    return img_gray


def predict(inp):
    img = inp['composite']/255
    print(f'img {img.shape} min {img.min()} max {img.max()}')
    # img = np.resize(img, (28, 28))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize((28,28)),
        transforms.Normalize((0,), (1,))  
    ])
    tensor = preprocess(img)
    print(f'tensor {tensor.shape}')
    print(f'values {tensor}')
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        out = model(tensor)
        pred = out.argmax()
        print(f'out {out} pred {pred}')
    return pred.item()
    

with gr.Blocks(title="Digit Classifier") as demo:
    gr.Markdown("## ✏️ Draw a Digit Below")

    sketchpad = gr.Sketchpad(image_mode='L', label="Draw a Digit a khawa")
    output = gr.Label(num_top_classes=1)

    btn = gr.Button("Predict")
    btn.click(fn=predict, inputs=sketchpad, outputs=output)

if __name__ == '__main__':
    demo.launch()
