import os
import torch
from torch.utils.data import Dataset
from skimage import io
import pandas as pd


class AdulterantDataset(Dataset):

    def __init__(self, dataset_path, transform=None):
        self.root_dir = dataset_path

        csv_file_path = os.path.join(self.root_dir, "concentration_data.csv")
        if not os.path.exists(csv_file_path):
            raise ValueError("CSV file not found")
        self.concentrations = pd.read_csv(csv_file_path)

        self.transform = transform

    def __len__(self):
        return len(self.concentrations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.concentrations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(self.concentrations.iloc[index, 1])

        if self.transform:
            image = self.transform(image)

        return (image, y_label)