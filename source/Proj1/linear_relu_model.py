import torch.nn as nn

""" Basic model with one hidden layer and two output units:
"""


class BaselineNet(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 2 * 14 * 14
        hidden_sizes = [392, 392]
        # two digit output, probability of being 1 or 0:
        output_size = 2
        # flatten images to 1D input:
        self.flatten = nn.Flatten()
        # then two hidden layers:
        # no need to add softmax at the end because already in CE loss.
        self.model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                   nn.ReLU(),
                                   nn.Linear(hidden_sizes[1], output_size))

    def forward(self, x):
        # flatten 2D->1D
        x = self.flatten(x)
        # predict probabilities:
        logits = self.model(x)

        return logits
