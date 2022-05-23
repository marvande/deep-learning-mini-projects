import torch
import torch.nn as nn


class SiameseNoSharingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 1 * 14 * 14
        hidden_sizes = [50, 50]
        # we need an intermediate output,
        # because we are using a siamese network
        output_size_1 = 2
        # two digit output, probability of being 1 or 0:
        output_size_2 = 2
        # flatten images to 1D input:
        self.flatten = nn.Flatten()
        # then two hidden layers:
        self.fc_seq2 = nn.Sequential(nn.Linear(self.input_size, hidden_sizes[0], bias=False),
                                     nn.ReLU(),
                                     nn.Linear(
                                         hidden_sizes[0], hidden_sizes[1], bias=False),
                                     nn.ReLU(),
                                     nn.Linear(hidden_sizes[1], output_size_1, bias=False))
        self.fc_seq1 = nn.Sequential(nn.Linear(self.input_size, hidden_sizes[0], bias=False),
                                     nn.ReLU(),
                                     nn.Linear(
                                         hidden_sizes[0], hidden_sizes[1], bias=False),
                                     nn.ReLU(),
                                     nn.Linear(hidden_sizes[1], output_size_1, bias=False))
        self.fcout = nn.Linear(2*output_size_1, output_size_2)

    def forward(self, x):
        x1 = x[:, 0].view(-1, self.input_size)
        x2 = x[:, 1].view(-1, self.input_size)
        output1 = self.fc_seq1(x1)
        output2 = self.fc_seq2(x2)
        output = torch.cat((output1, output2), 1)
        # flatten 2D->1D
        output = self.flatten(output)
        # predict probabilities:
        logits = self.fcout(output)

        return logits
