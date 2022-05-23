import torch
import torch.nn as nn


class SiameseConvNoSharingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.length = 14
        self.input_size = 1 * 14 * 14
        # two digit output, probability of being 1 or 0:
        output_size = 2
        # flatten images to 1D input:
        self.flatten = nn.Flatten()
        # convolutional layers
        k = 3  # kernel size
        c = 3  # number of channels
        self.conv_layer1 = nn.Sequential(nn.Conv2d(1, c, k),
                                         nn.ReLU(),
                                         nn.Conv2d(c, 2 * c, k),
                                         nn.ReLU(),
                                         nn.Conv2d(2 * c, 4 * c, k),
                                         nn.ReLU(),
                                         nn.Conv2d(4 * c, 8 * c, k),
                                         nn.ReLU())
        self.conv_layer2 = nn.Sequential(nn.Conv2d(1, c, k),
                                         nn.ReLU(),
                                         nn.Conv2d(c, 2 * c, k),
                                         nn.ReLU(),
                                         nn.Conv2d(2 * c, 4 * c, k),
                                         nn.ReLU(),
                                         nn.Conv2d(4 * c, 8 * c, k),
                                         nn.ReLU()
                                         )

        def compute_conv2d_size(length):
            return length - (k - 1) - 1 + 1

        length_out = self.length
        depth = 4

        for i in range(depth):
            length_out = compute_conv2d_size(length_out)

        concat_size = length_out * length_out * 2**depth * c
        self.fcout = nn.Linear(concat_size, output_size)

    def forward(self, x):
        x1 = x[:, 0].view(-1, 1, self.length, self.length)
        x2 = x[:, 1].view(-1, 1, self.length, self.length)
        output1 = self.conv_layer1(x1)
        output2 = self.conv_layer2(x2)
        output = torch.cat((output1, output2), 1)
        # flatten 2D->1D
        output = self.flatten(output)
        # predict probabilities:
        logits = self.fcout(output)

        return logits
