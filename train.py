from PureMilk.model import AdulterantDetector
from PureMilk.dataset import AdulterantDataset
import os

model = AdulterantDetector([100, 100, 1])

parent_dir = os.path.dirname(os.path.abspath(''))
dataset_path = os.path.join(parent_dir, 'drive', 'My Drive', 'data', 'NaOH')
datatset = AdulterantDataset(dataset_path)