from PureMilk.model import AdulterantDetector
from PureMilk.dataset import AdulterantDataset
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# params
learning_rate = 0.001
batch_size = 4
num_epochs = 10

dataset_path = sys.argv[1]
if not os.path.exists(dataset_path):
    raise ValueError("Dataset path not found")
train_dataset = AdulterantDataset(dataset_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = AdulterantDetector([256*3, 128, 128, 1]).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in tqdm(range(num_epochs)):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        data = torch.reshape(data, (data.shape[0], -1))
        target = torch.reshape(target, (target.shape[0], 1))

        preds = model(data)

        loss = criterion(preds, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    # if epoch % 20 == 0:
    #     utils.draw_non_blocking(utils.predict_image(model, image_resolution))

# utils.check_accuracy(train_loader, model)
# utils.draw(utils.predict_image(model))