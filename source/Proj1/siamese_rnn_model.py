import torch
import torch.nn as nn


class ImageRNN(nn.Module):

    # declaraction of variables
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs):
        super(ImageRNN, self).__init__()

        # we need an intermediate output,
        # because we are using a siamese RNN network
        intermediate_output_size = 10

        # two digit output, probability of being 1 or 0:
        output_size = 2

        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # flatten images to 1D input:
        self.flatten = nn.Flatten()

        self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons,  bias=False)

        self.FC = nn.Linear(
            self.n_neurons, intermediate_output_size, bias=False)
        self.fcout = nn.Linear(2*intermediate_output_size,
                               output_size,  bias=False)

    # initialize hidden weights that have zero values

    def init_hidden(self, ):
        # (num_layers, batch_size, n_neurons)
        return (torch.zeros(1, self.batch_size, self.n_neurons))

    def forward(self, X):
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        X1 = X[:, 0, :].view(-1, 14, 14).permute(1, 0, 2)
        X2 = X[:, 1, :].view(-1, 14, 14).permute(1, 0, 2)
        # X1.shape = torch.Size([14, 64, 14])

        # ------------------------------------------------------------------
        # Input image 1 (first image of the pair)
        # Predict the number on first image
        # ------------------------------------------------------------------
        self.batch_size = X1.size(1)
        self.hidden = self.init_hidden()

        # lstm_out => n_steps, batch_size, n_neurons (hidden states for each time step)
        # self.hidden => 1, batch_size, n_neurons (final state from each lstm_out)
        lstm_out, self.hidden = self.basic_rnn(X1, self.hidden)

        # lstm_out.shape = torch.Size([14, 64, 50])
        # self.hidden.shape = torch.Size([1, 64, 50])

        # Size batch x 10 (cipher predictions on image)
        out1 = self.FC(self.hidden)
        # out1.shape = torch.Size([1, 64, 10])

        # ------------------------------------------------------------------
        # Input image 2 (second image of the pair)
        # Predict the number on second image
        # ------------------------------------------------------------------
        self.batch_size = X2.size(1)
        self.hidden = self.init_hidden()

        # lstm_out => n_steps, batch_size, n_neurons (hidden states for each time step)
        # self.hidden => 1, batch_size, n_neurons (final state from each lstm_out)
        lstm_out, self.hidden = self.basic_rnn(X2, self.hidden)

        # Size batch x 10 (cipher predictions on image)
        out2 = self.FC(self.hidden)

        # ------------------------------------------------------------------
        # 0-1 Prediction
        # Predict if first image bigger than second
        # ------------------------------------------------------------------

        # concatenate into size (batch x 20)
        output = torch.cat((out1[0], out2[0]), 1)
        # output.shape = torch.Size([64, 20])

        # flatten 2D->1D
        output_ = self.flatten(output)
        # output_.shape = torch.Size([64, 20])

        # predict probabilities:
        logits = self.fcout(output_)

        return out1.view(-1, self.n_outputs), out2.view(
            -1, self.n_outputs), logits
