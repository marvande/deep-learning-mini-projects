"""Accuracy, ciphers accuracy and weighted loss evaluator

    This module allows the user to evaluate the accuracy, ciphers accuracy
    and weighted loss of a given deep learning model and to save the plot as png.

    The model must be compatible and return (pred_cipher_1, pred_cipher_2, w_loss): 
    it must return the prediction of digits (for the two digits) and the weighted loss. 

        * plot_performance - create and save the plot of the evaluation as png.
        * train_loop - optimize the model and store information for evaluation
        * test_loop - store information for evaluation and use the test set
        * train_eval - generate the dataset, train and test the model 
        * evaluate - instantiate, train, evaluate the model and save the plot as png
"""

import os
import torch
from dataset import Dataset
import torch.nn as nn
import matplotlib.pyplot as plt
from dlc_practical_prologue import *


def plot_performance(train_perf, test_perf, model_name):
    """Create and save the plot of the accuracy, ciphers accuracy and weighted loss during training and testing.

    Args:
        train_perf (list(tuple)): training data comprising the accuracy, ciphers accuracy and weighted loss training
        test_perf (list(tuple)): test data comprising the accuracy, ciphers accuracy and weighted loss during test
        model_name (str): name of the model
    """
    def sub_plot(axs_id, train_data, test_data, train_label, test_label, x_label, title):
        axs[axs_id].plot(train_data, label=train_label)
        axs[axs_id].plot(test_data, label=test_label)
        axs[axs_id].set_xlabel(x_label)
        axs[axs_id].set_title(title)
        axs[axs_id].legend()

    fig, axs = plt.subplots(1, 3, figsize=(8, 4))

    train_accs = list(zip(*train_perf))[0]
    test_accs = list(zip(*test_perf))[0]
    sub_plot(0, train_accs, test_accs, 'train accuracy',
             'test accuracy', 'Num epochs', 'Accuracy')

    # Accuracy on ciphers:
    test_accs_ciphers = list(zip(*test_perf))[1]
    train_accs_ciphers = list(zip(*train_perf))[1]
    sub_plot(1, train_accs_ciphers,  test_accs_ciphers,
             'train accuracy', 'test accuracy', 'Num epochs', 'Accuracy ciphers')

    train_losses = list(zip(*train_perf))[2]
    test_losses = list(zip(*test_perf))[2]
    sub_plot(2, train_losses, test_losses, 'train loss',
             'test loss', 'Num epochs', 'Loss')

    plt.tight_layout()

    # save plot
    filename = "{}_plot.png".format(model_name)
    path = "plots/" + filename
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    print("plot saved as {} in plots folder!".format(filename))


def train_loop(dataloader, model, loss_fn, optimizer, loss_weight, print_loss=False):
    """Train the model and store accuracy, ciphers accuracy and weighted loss information for evaluation.

    Args:
        dataloader (Dataloader): data to train the model
        model (Module): instantiated model to optimize
        loss_fn (Loss): loss module to be used
        optimizer (Optimizer): optimizer to be used
        loss_weight (list): weights to compute the weighted loss
        print_loss (bool, optional): print the loss during training if True. Defaults to False.

    Returns:
        tuple: accuracy, ciphers accuracy and weighted loss
    """
    size = len(dataloader.dataset)
    train_loss, accuracy, accuracy_numbers = 0, 0, 0
    for batch, (X, y, Y) in enumerate(dataloader):
        # Compute prediction and loss:
        pred1, pred2, logits = model(X)

        # Softmax to get probabilities:
        prob = torch.nn.Softmax(dim=1)(logits)

        # calculate number of correct predictions:
        accuracy += (prob.argmax(1) == y).type(torch.float).sum().item()
        accuracy_numbers += (pred1.argmax(1) ==
                             Y[:, 0]).type(torch.float).sum().item()
        accuracy_numbers += (pred2.argmax(1) ==
                             Y[:, 1]).type(torch.float).sum().item()

        # [0-1] pred loss:
        loss = loss_fn(logits, y)

        # [0-9] pred loss for each pair:
        loss_aux_1 = loss_fn(pred1, Y[:, 0])
        loss_aux_2 = loss_fn(pred2, Y[:, 1])

        # Backpropagation:
        optimizer.zero_grad()
        loss = loss_weight[0]*loss + loss_weight[1] * loss_aux_1 + loss_weight[2] * \
            loss_aux_2  # 0.4 is weight for auxillary classifier

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        loss, current = loss.item(), batch * len(X)

        if print_loss:
            if batch % 10 == 0:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # return average training loss:
    train_loss /= size
    accuracy *= 100 / size
    accuracy_numbers *= 100 / (2 * size)

    return accuracy, accuracy_numbers, train_loss


def test_loop(dataloader, model, loss_fn, loss_weight):
    """Train the model and store accuracy, ciphers accuracy and weighted loss information for evaluation.

    Args:
        dataloader (Dataloader): data to test the model
        model (Module): trained model to test
        loss_fn (Loss): loss module to be used
        loss_weight (list): weights to compute the weighted loss

    Returns:
        tuple: accuracy, ciphers accuracy and weighted loss
    """
    size = len(dataloader.dataset)
    test_loss, accuracy, accuracy_numbers = 0, 0, 0
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for X, y, Y in dataloader:
            pred1, pred2, logits = model(X)

            # [0-1] pred loss:
            loss = loss_fn(logits, y)

            # [0-9] pred loss for each pair:
            loss_aux_1 = loss_fn(pred1, Y[:, 0])
            loss_aux_2 = loss_fn(pred2, Y[:, 1])

            # sum with weights for total loss:
            loss = loss_weight[0]*loss + loss_weight[1] * \
                loss_aux_1 + loss_weight[2] * loss_aux_2
            test_loss += loss.item()

            # Softmax to get probabilities:
            prob = softmax(logits)

            # calculate number of correct predictions:
            accuracy += (prob.argmax(1) == y).type(torch.float).sum().item()
            accuracy_numbers += (pred1.argmax(1) ==
                                 Y[:, 0]).type(torch.float).sum().item()
            accuracy_numbers += (pred2.argmax(1) ==
                                 Y[:, 1]).type(torch.float).sum().item()

    # return average test loss and accuracy:
    test_loss /= size
    accuracy *= 100 / size
    accuracy_numbers *= 100 / (2 * size)

    return accuracy, accuracy_numbers, test_loss


def train_eval(model, optimizer, loss_fn, epochs=25, save=False, print_loss=False, print_epoch=False):
    """Train and test the given model and returns evaluation data ready for creating the plot.

    Args:
        model (Module): instantiated model to optimize and test
        optimizer (Optimizer): optimizer to train the model
        loss_fn (Loss): loss module to be used
        epochs (int, optional): number of epochs. Defaults to 25.
        save (bool, optional): save the model and the evaluation data if True. Defaults to False.
        print_loss (bool, optional): print the loss during training and test if True. Defaults to False.
        print_epoch (bool, optional): print the epochs if True. Defaults to False.

    Returns:
        list(tuple), list(tuple): training (and test) accuracy, ciphers accuracy and weighted loss
    """
    print("Training and testing the model with {} epochs.".format(epochs))

    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(
        1000)

    training_set = Dataset(train_input, train_target, train_classes)
    test_set = Dataset(test_input, test_target, test_classes)

    # Data loader for model, change num_workers when on GPU:
    params = {'batch_size': 64, 'shuffle': True, 'num_workers': 0}
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    test_generator = torch.utils.data.DataLoader(test_set, **params)
    train_perf, test_perf = [], []
    loss_weight = [1, 0.8, 0.8]

    for t in range(epochs):
        if print_epoch:
            print(f"Epoch {t+1}\n----------")
        train_perf.append(train_loop(training_generator,
                                     model,
                                     loss_fn,
                                     optimizer,
                                     loss_weight,
                                     print_loss=print_loss))
        test_perf.append(test_loop(test_generator,
                                   model,
                                   loss_fn,
                                   loss_weight))

    print("Done!")

    if save:
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_perf': train_perf,
                    'test_perf': test_perf
                    }, '../data/{}'.format(type(model).__name__))
        print("Saved!")

    return train_perf, test_perf


def evaluate(model, epochs):
    """Evaluate the given model using accuracy, ciphers accuracy and weighted loss,
       then save the performance plot as png.

    Args:
        model (Module): instantiated model to be evaluated
        epochs (int): number of epochs of evaluation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------------------
    # Control the randomness
    # ------------------------------------------------------------------
    torch.manual_seed(0)

    # ------------------------------------------------------------------
    # Initialize/reset the model and optimizer (i.e. reset the weights)
    # ------------------------------------------------------------------
    model_name = type(model).__name__
    print("Instantiating {} model.".format(model_name))
    model = model.to(device)
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ------------------------------------------------------------------
    # Train and evaluate
    # ------------------------------------------------------------------
    loss_fn = nn.CrossEntropyLoss()
    train_perf, test_perf = train_eval(model,
                                       optimizer,
                                       loss_fn=loss_fn,
                                       epochs=epochs,
                                       print_loss=False,
                                       print_epoch=True)

    # ------------------------------------------------------------------
    # Plot performance
    # ------------------------------------------------------------------
    plot_performance(train_perf, test_perf, model_name)
