import os
import torch
from torch.utils.data import Dataset
from skimage import io
import pandas as pd
from skimage.transform import resize

class AdulterantDataset(Dataset):

    def __init__(self, dataset_path: str, transform=None):
        self.root_dir = dataset_path

        csv_file_path = os.path.join(self.root_dir, "data.csv")
        if not os.path.exists(csv_file_path):
            raise ValueError("CSV file not found")
        self.annotations = pd.read_csv(csv_file_path)

        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)[:,:,:3]
        image = resize(image, (16, 16))
        image = torch.tensor(image, dtype=torch.float32)
        y_label = torch.tensor(self.annotations.iloc[index, 1], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return (image, y_label)