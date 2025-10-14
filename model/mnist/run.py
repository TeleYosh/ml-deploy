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


def predict(inp):
    # print(f'img {img.keys()} ')
    # print(f'type {type(img['background'])}')
    # print(f'type {type(img['layers'])}')
    # print(f'type {type(img['composite'])}')
    # print('batck', img['background'].shape)
    # print(len(img['layers']))
    # print('cmp', img['composite'].shape)
    img = inp['composite']
    img = np.resize(img, (28, 28))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))  
    ])
    tensor = preprocess(img)
    print('tensor', tensor)
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        out = model(tensor)
        print('out', out)
    return int(torch.argmax(out))

with gr.Blocks(title="Digit Classifier") as demo:
    gr.Markdown("## ✏️ Draw a Digit Below")

    sketchpad = gr.Sketchpad(crop_size=(28, 28), label="Draw a Digit a khawa dialo")
    output = gr.Label(num_top_classes=1)

    btn = gr.Button("Predict")
    btn.click(fn=predict, inputs=sketchpad, outputs=output)

if __name__ == '__main__':
    demo.launch()
