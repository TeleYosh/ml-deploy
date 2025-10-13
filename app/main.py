import gradio as gr
import numpy as np
# import io

def main():
    # Load your model
    # model = torch.load("model/model.pth", map_location="cpu")
    # model.eval()

    def predict(img):
        # img = img.convert("L").resize((28, 28))
        # arr = np.array(img) / 255.0
        # tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).float()
        # with torch.no_grad():
        #     out = model(tensor)
        # return int(torch.argmax(out))
        return np.random.randint(0, 10)

    iface = gr.Interface(
        fn=predict,
        inputs=gr.Sketchpad(crop_size=(256,256), label="Draw a Digit a khawa"),
        outputs=gr.Label(num_top_classes=1),
        title="Digit Classifier"
    )

    # iface.launch(server_name="0.0.0.0", server_port=8000)
    iface.launch(server_name='0.0.0.0', root_path="/gradio")

if __name__ == "__main__":
    main()
