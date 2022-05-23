import torch
import torch.nn as nn


class SiameseConvNet2(nn.Module):
    def __init__(self):
        super(SiameseConvNet2, self).__init__()
        self.length = 14
        self.input_size = 1 * 14 * 14
        # we need an intermediate output,
        # because we are using a siamese network
        intermediate_output_size = 10
        self.n_outputs = 10
        # two digit output, probability of being 1 or 0:
        final_output_size = 2

        # flatten images to 1D input:
        self.flatten = nn.Flatten()
        # convolutional layers
        kernel_size = 3
        n_channel = 5
        self.conv_layer = nn.Sequential(nn.Conv2d(1, n_channel, kernel_size),
                                        nn.ReLU(),
                                        nn.Conv2d(n_channel, 2 *
                                                  n_channel, kernel_size),
                                        nn.ReLU(),
                                        nn.Conv2d(2 * n_channel, 4 *
                                                  n_channel, kernel_size),
                                        nn.BatchNorm2d(4 * n_channel),
                                        nn.ReLU(),
                                        nn.Dropout2d(p=0.3),
                                        nn.Conv2d(4 * n_channel, 8 *
                                                  n_channel, kernel_size),
                                        nn.ReLU(),
                                        nn.Dropout2d(p=0.4),
                                        nn.Conv2d(8 * n_channel, 16 *
                                                  n_channel, kernel_size),
                                        nn.ReLU(),
                                        nn.Dropout2d(p=0.5),
                                        nn.Conv2d(16 * n_channel,
                                                  32 * n_channel, kernel_size),
                                        nn.ReLU(),
                                        nn.Dropout2d(p=0.6)
                                        )

        def compute_conv2d_size(length):
            return length - (kernel_size - 1) - 1 + 1

        length_out = self.length
        depth = 4

        for i in range(depth):
            length_out = compute_conv2d_size(length_out)

        # For 10 digits:
        concat_size = 640
        self.fcout1 = nn.Linear(concat_size, intermediate_output_size)

        # For 0-1 output:
        self.fcout2 = nn.Linear(2 * intermediate_output_size,
                                final_output_size,
                                bias=False)

    def forward(self, x):
        x1 = x[:, 0].view(-1, 1, self.length, self.length)
        x2 = x[:, 1].view(-1, 1, self.length, self.length)
        # x1.shape = torch.Size([64, 1, 14, 14])

        x1 = self.conv_layer(x1)
        # x1.shape = torch.Size([64, 160, 2, 2])
        x1 = x1.view(-1, 640)
        # x1.shape = torch.Size([64, 640])

        # Size batch x 10 (cipher predictions on image)
        out1 = self.fcout1(x1)
        # out1.shape = torch.Size([64, 10])

        x2 = self.conv_layer(x2)
        x2 = x2.view(-1, 640)
        # Size batch x 10 (cipher predictions on image)
        out2 = self.fcout1(x2)

        # ------------------------------------------------------------------
        # 0-1 Prediction
        # Predict if first image bigger than second
        # ------------------------------------------------------------------

        # concatenate into size (batch x 20)
        output = torch.cat((out1, out2), 1)
        # predict probabilities:
        logits = self.fcout2(output)

        return out1.view(-1, self.n_outputs), out2.view(-1,
                                                        self.n_outputs), logits
