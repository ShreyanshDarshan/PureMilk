import torch
from torch.utils.data import DataLoader
from PureMilk.model import AdulterantDetector
from PureMilk.dataset import AdulterantDataset
from PureMilk.utils import check_accuracy
from PureMilk.train import train_model
import os
import sys


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # params
    learning_rate = 0.001
    batch_size = 4
    num_epochs = 100

    dataset_path = sys.argv[1]
    if not os.path.exists(dataset_path):
        raise ValueError("Dataset path not found")
    train_dataset = AdulterantDataset(dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = AdulterantDetector([3, 6, 12], [12*8*8, 128, 128, 1]).to(device)
    train_model(train_loader, model, learning_rate, num_epochs)

    check_accuracy(train_loader, model)