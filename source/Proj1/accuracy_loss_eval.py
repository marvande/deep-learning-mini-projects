"""Accuracy and Loss evaluator

    This module allows the user to evaluate the accuracy and loss of
    a given deep learning model and to save the plot as png.

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
    """Create and save the plot of the accuracy and loss during training and testing.

    Args:
        train_perf (list(tuple)): the training data comprising the accuracy and loss during training
        test_perf (list(tuple)): the test data comprising the accuracy and loss during test
        model_name (str): the name of the model
    """
    def sub_plot(axs_id, train_data, test_data, train_label, test_label, x_label, title):
        axs[axs_id].plot(train_data, label=train_label)
        axs[axs_id].plot(test_data, label=test_label)
        axs[axs_id].set_xlabel(x_label)
        axs[axs_id].set_title(title)
        axs[axs_id].legend()

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # Accuracy subplot
    train_accs = list(zip(*train_perf))[0]
    test_accs = list(zip(*test_perf))[0]
    sub_plot(0, train_accs, test_accs, 'train accuracy',
             'test accuracy', 'Num epochs', 'Accuracy')

    # Loss subplot
    train_losses = list(zip(*train_perf))[1]
    test_losses = list(zip(*test_perf))[1]
    sub_plot(1, train_losses, test_losses, 'train loss',
             'test loss', 'Num epochs', 'Loss')

    plt.tight_layout()

    # save plot
    filename = "{}_plot.png".format(model_name)
    path = "plots/" + filename
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    print("plot saved as {} in plots folder!".format(filename))


def train_loop(dataloader, model, loss_fn, optimizer, print_loss=False):
    """Train the model and store accuracy and loss information for evaluation.

    Args:
        dataloader (Dataloader): data to train the model
        model (Module): instantiated model to optimize
        loss_fn (Loss): loss module to be used
        optimizer (Optimizer): optimizer to be used
        print_loss (bool, optional): print the loss during training if True. Defaults to False.

    Returns:
        tuple: the accuracy and loss
    """
    size = len(dataloader.dataset)
    train_loss, accuracy = 0, 0
    softmax = torch.nn.Softmax(dim=1)

    for batch, (X, y, Y) in enumerate(dataloader):
        # Compute prediction and loss:
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if print_loss:
            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # Softmax to get probabilities:
        prob = softmax(pred)

        # compute accuracy :
        accuracy += (prob.argmax(1) == y).type(torch.float).sum().item()

    # return average training loss:
    train_loss /= size
    accuracy *= 100 / size

    return accuracy, train_loss


def test_loop(dataloader, model, loss_fn, print_loss=False):
    """Test the model and store accuracy and loss information for evaluation.

    Args:
        dataloader (Dataloader): data to test the model
        model (Module): trained model to test
        loss_fn (Loss): loss module to be used
        print_loss (bool, optional): print the loss during test if True. Defaults to False.

    Returns:
        tuple: the accuracy and loss
    """
    size = len(dataloader.dataset)
    test_loss, accuracy = 0, 0
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for X, y, Y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # Softmax to get probabilities:
            prob = softmax(pred)

            # compute accuracy:
            accuracy += (prob.argmax(1) == y).type(torch.float).sum().item()

    # return average test loss and accuracy:
    test_loss /= size
    accuracy *= 100 / size

    if print_loss:
        print(
            f"Validation Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )

    return accuracy, test_loss


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
        list(tuple), list(tuple): training (and test) accuracy and loss
    """
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(
        1000)

    print("Training and testing the model with {} epochs.".format(epochs))

    training_set = Dataset(train_input, train_target, train_classes)
    test_set = Dataset(test_input, test_target, test_classes)

    # Data loader for model, change num_workers when on GPU:
    params = {'batch_size': 64, 'shuffle': True, 'num_workers': 0}
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    test_generator = torch.utils.data.DataLoader(test_set, **params)
    train_perf, test_perf = [], []

    for t in range(epochs):
        if print_epoch:
            print(f"Epoch {t+1}\n----------")
        train_perf.append(train_loop(training_generator,
                                     model,
                                     loss_fn,
                                     optimizer,
                                     print_loss=print_loss))
        test_perf.append(test_loop(test_generator,
                                   model,
                                   loss_fn,
                                   print_loss=print_loss))

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
    """Evaluate the given model using accuracy and loss, then save the performance plot as png.

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
