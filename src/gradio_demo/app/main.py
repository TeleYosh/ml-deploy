import gradio as gr
import numpy as np
import torch 
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
model = torch.load("../model/sketch_images/weights/cnn.pth", map_location="cpu")
model.eval()

# def predict(img):
#     # img = img.convert("L").resize((28, 28))
#     # arr = np.array(img) / 255.0
#     # tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).float()
#     # with torch.no_grad():
#     #     out = model(tensor)
#     # return int(torch.argmax(out))
#     return 'djg'

# demo = gr.Interface(
#     fn=predict,
#     inputs=gr.Sketchpad(crop_size=(256,256), label="Draw a Digit a khawa dialo"),
#     outputs=gr.Label(num_top_classes=1),
#     title="Digit Classifier"
# )

# # iface.launch(server_name="0.0.0.0", server_port=8000)
# # iface.launch(server_name='0.0.0.0', root_path="/gradio")
# # iface.launch()

if __name__ == "__main__":
    # gr.serve(demo, reload=True)
    # demo.launch()
    a = torch.rand(size=(1, 1, 28, 28))
    out = model(a)
    print(out)
