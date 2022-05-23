import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_basic(nn.Module):
    # declaraction of variables
    def __init__(self):
        super(Conv_basic, self).__init__()

        # we need an intermediate output,
        # because we are using a siamese RNN network
        intermediate_output_size = 10

        # two digit output, probability of being 1 or 0:
        fina_output_size = 2
        self.n_outputs = 10
        # flatten images to 1D input:
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(40, 30)
        self.fc2 = nn.Linear(30, 10)

        self.fcout = nn.Linear(2 * intermediate_output_size,
                               fina_output_size,
                               bias=False)

    def forward(self, X):
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        X1 = X[:, 0, :].view(-1, 1, 14, 14)
        X2 = X[:, 1, :].view(-1, 1, 14, 14)
        # X1.shape = torch.Size([64, 1, 14, 14])

        # ------------------------------------------------------------------
        # Input image 1 (first image of the pair)
        # Predict the number on first image
        # ------------------------------------------------------------------
        x = F.relu(F.max_pool2d(self.conv1(X1), 2))
        # x.shape = torch.Size([64, 10, 5, 5])
        x = F.relu(F.max_pool2d(x, 2))
        # x.shape = torch.Size([64, 10, 2, 2])
        x = x.view(-1, 40)
        # x.shape = torch.Size([64, 40])
        x = F.relu(self.fc1(x))
        # x.shape = torch.Size([64, 30])
        x = F.dropout(x, training=self.training)
        # x.shape = torch.Size([64, 30])

        # Size batch x 10 (cipher predictions on image)
        out1 = self.fc2(x)
        # out1.shape = torch.Size([64, 10])

        # ------------------------------------------------------------------
        # Input image 2 (second image of the pair)
        # Predict the number on second image
        # ------------------------------------------------------------------
        x = F.relu(F.max_pool2d(self.conv1(X2), 2))
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 40)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        # Size batch x 10 (cipher predictions on image)
        out2 = self.fc2(x)

        # ------------------------------------------------------------------
        # 0-1 Prediction
        # Predict if first image bigger than second
        # ------------------------------------------------------------------

        # concatenate into size (batch x 20)
        output = torch.cat((out1, out2), 1)
        # flatten 2D->1D
        output_ = self.flatten(output)

        # predict probabilities:
        logits = self.fcout(output_)

        return out1.view(-1, self.n_outputs), out2.view(-1,
                                                        self.n_outputs), logits
