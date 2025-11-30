from sketch_recognition.src.models.cnn import CNN, CNN_v2
from sketch_recognition.src.models.utils import get_validation_metrics
import json
import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_dataset
from torchvision import transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAINING_DIR = Path(__file__).resolve().parent 
SRC_DIR = TRAINING_DIR.parent  
PARAMS_PATH = SRC_DIR / 'models' / 'params' / 'cnn_45_params.json'
WEIGHTS_PATH = SRC_DIR / 'models' / 'weights' / 'cnn_45_params.json'
with open(PARAMS_PATH, 'r') as f:
    params = json.load(f)

cnn = CNN(**params, n_classes=345)
cnn.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
cnn.eval()

dataset = load_dataset("Xenova/quickdraw-small")
preprocess = transforms.Compose([
    transforms.ToTensor(), 
])
def preprocess_ops(examples):
    examples['image'] = [preprocess(image) for image in examples['image']]
    return examples
dataset.set_transform(preprocess_ops)
val_dataset = dataset['validation']
batch_size = 256
criterion = nn.CrossEntropyLoss()

metrics = get_validation_metrics(cnn, val_dataset, batch_size, criterion, device)
print(f'acc {metrics['accuracy']}, loss {metrics['loss']}, f1_score {metrics['f1_score']}')
