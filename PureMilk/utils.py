import torch
from torch.utils.data import DataLoader
import numpy as np

def check_accuracy(loader: DataLoader, model: torch.nn.Module):
    device = torch.device(
        'cuda' if next(model.parameters()).is_cuda else 'cpu')

    model.eval()
    with torch.no_grad():
        errors = []
        for batch_idx, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)

            data = torch.reshape(data, (data.shape[0], -1))
            target = torch.reshape(target, (target.shape[0], 1))

            preds = model(data)
            errors += torch.abs(preds - target).cpu().tolist()

        mean_error = np.mean(errors)
        std_dev = np.std(errors)

    print('Mean Error:         %.2f%%', (mean_error))
    print('Standard Deviation: %.2f%%', (std_dev))