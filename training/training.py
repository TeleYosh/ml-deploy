import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
import json
from cnn import CNN
from utils import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Current device is: {device}')

# data loading and preprocessing
dataset = load_dataset("Xenova/quickdraw-small")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0,), (1,))  
])
def preprocess_ops(examples):
    examples['image'] = [preprocess(image) for image in examples['image']]
    return examples
dataset.set_transform(preprocess_ops)

train_dataset, test_dataset, val_dataset = dataset['train'], dataset['test'], dataset['valid']
train_dataset = train_dataset.shard(num_shards=20, index=0)
names = train_dataset.features['label'].names
n_classes = len(names)
print(f'size of trainset: {len(train_dataset)}, testset: {len(test_dataset)}')

# get labels
with open('labels.json', 'w') as f:
    json.dump(names, f)
def id_to_class(idx):
    return names[idx]

# model loading
params = {
    'n_filters': 30,
    'hidden_dim': 100,
    'n_layers': 2,
    'n_classes': n_classes
}
model = CNN(**params).to(device)
n_params = sum([p.numel() for p in model.parameters()])
print(f'Number of params {n_params}')

# hyperparameters
lr = 0.01
batch_size = 128*2
n_epochs = 5

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

trainLoader = DataLoader(train_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         pin_memory=True,
                         num_workers=1,
                         prefetch_factor=2,       # Prepare batches ahead of time
                         persistent_workers=True)
testLoader = DataLoader(test_dataset, batch_size=batch_size)

train_losses, train_accs, test_losses, test_accs = train(
                                                        model,
                                                        trainLoader,
                                                        testLoader,
                                                        criterion,
                                                        optimizer,
                                                        n_epochs,
                                                        device)