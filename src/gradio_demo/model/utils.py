from typing import Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torch import Tensor
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import math as m

def train(model, trainLoader, testLoader, criterion, optimizer, n_epochs, device):
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    for epoch in range(1, n_epochs+1):
        model.train()
        train_loss = 0
        train_acc = 0
        for batch in tqdm(trainLoader):
            data, labels = batch['image'].to(device), batch['label'].to(device)
            out = model(data)
            preds = out.argmax(dim=1)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (preds == labels).sum().item()

        train_loss /= len(trainLoader)
        train_acc /= len(trainLoader.dataset)
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0
        test_acc = 0
        with torch.no_grad():
            for batch in tqdm(testLoader, disable=True):
                data, labels = batch['image'].to(device), batch['label'].to(device)
                out = model(data)
                loss = criterion(out, labels)
                preds = out.argmax(dim=1)
                test_loss += loss.item()
                test_acc += (preds == labels).sum().item()

        test_loss /= len(testLoader)
        test_acc /= len(testLoader.dataset)
        test_accs.append(test_acc)
        test_losses.append(test_loss)

        print(f'epoch {epoch} | train loss {train_loss:.3f} train acc {train_acc:.2f} | test loss {test_loss:.3f} test acc {test_acc:.2f}')
    return train_losses, train_accs, test_losses, test_accs


def get_validation_metrics(
    model: Module,
    val_dataset: Dataset,
    batch_size: int,
    criterion: Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate a PyTorch model on a validation dataset and compute key performance metrics.

    This function runs the model in evaluation mode on the given validation dataset,
    computing the average loss, accuracy, precision, recall, and F1-score for a
    multiclass classification task.

    Args:
        model (torch.nn.Module): The trained PyTorch model to evaluate.
        val_dataset (torch.utils.data.Dataset): Validation dataset containing samples
            with 'image' and 'label' keys.
        batch_size (int): Number of samples per batch during evaluation.
        criterion (torch.nn.Module): Loss function used to compute validation loss.
        device (torch.device): Device on which to perform computation (e.g., 'cuda' or 'cpu').

    Returns:
        Dict[str, float]: A dictionary containing:
            - 'loss': Average validation loss.
            - 'accuracy': Overall accuracy across the validation dataset.
            - 'precision': Weighted average precision across classes.
            - 'recall': Weighted average recall across classes.
            - 'f1_score': Weighted average F1-score across classes.
    """
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    all_preds, all_labels = [], []
    val_loss, val_acc = 0.0, 0.0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            data: Tensor = batch['image'].to(device)
            labels: Tensor = batch['label'].to(device)

            outputs: Tensor = model(data)
            loss: Tensor = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            val_loss += loss.item()
            val_acc += (preds == labels).sum().item()

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    val_loss /= len(val_loader)
    val_acc /= len(val_loader.dataset)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )

    metrics: Dict[str, float] = {
        'loss': val_loss,
        'accuracy': val_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }

    return metrics

def output_conv_size(height, width, conv_kernel_size,maxpool_kernel_size, n_filters):
    '''
    Give size of tensor after convolutions and maxpoolings.
    '''
    return 2*n_filters* m.floor((m.floor((height-conv_kernel_size+1-maxpool_kernel_size)/maxpool_kernel_size+1)-conv_kernel_size+1-maxpool_kernel_size)/maxpool_kernel_size+1)*m.floor((m.floor((width-conv_kernel_size+1-maxpool_kernel_size)/maxpool_kernel_size+1)-conv_kernel_size+1-maxpool_kernel_size)/maxpool_kernel_size+1)