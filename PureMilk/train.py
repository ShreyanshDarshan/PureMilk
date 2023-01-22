from PureMilk.model import AdulterantDetector
from PureMilk.dataset import AdulterantDataset
from PureMilk.utils import check_accuracy
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader


def train_model(loader: DataLoader, model: nn.Module, learning_rate: float=0.001, num_epochs: int=100):
    device = torch.device(
        'cuda' if next(model.parameters()).is_cuda else 'cpu')

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        for batch_idx, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)

            data = torch.reshape(data, (data.shape[0], 3, 32, 32))
            target = torch.reshape(target, (target.shape[0], 1))

            preds = model(data)

            loss = criterion(preds, target)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()