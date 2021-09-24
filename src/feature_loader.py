from torch.utils.data import Dataset
from numpy import load
from torch import tensor, float32, long


class DatasetFromArray(Dataset):
    def __init__(self, featurepath, labelpath):
        self.data = load(open(featurepath, "rb"))
        self.labels = load(open(labelpath, "rb"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "data": tensor(self.data[index], dtype=float32),
            "target": tensor(self.labels[index], dtype=long),
        }
