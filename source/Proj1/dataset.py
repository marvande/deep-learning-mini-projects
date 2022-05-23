import torch


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, pairs, target, classes):
        'Initialization'
        # target = (0,1)
        self.target = target
        # image pairs (2,14,14)
        self.pairs = pairs
        # cipher classes (2 in [0,9])
        self.classes = classes

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.pairs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # image pairs
        X = self.pairs[index]
        # target:
        y = self.target[index]
        # classes:
        Y = self.classes[index]
        return X, y, Y
