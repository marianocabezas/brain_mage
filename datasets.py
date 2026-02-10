import numpy as np
from torch.utils.data.dataset import Dataset


''' Datasets '''

class LongitudinalDataset(Dataset):
    def __init__(
        self, pairs, labels
    ):
        # Init
        self.pairs = pairs
        self.labels = labels

    def __getitem__(self, index):

        pair = self.pairs[index].astype(np.float32)
        mean =  np.mean(pair, axis=(1, 2, 3), keepdims=True)
        std = np.std(pair, axis=(1, 2, 3), keepdims=True)
        data = (pair - mean) / std
        label = self.labels[index]
        # Patch "extraction".
        target_data = np.expand_dims(
            np.array(label, dtype=np.uint8),
            axis=0
        )

        return data, target_data

    def __len__(self):
        return len(self.labels)
