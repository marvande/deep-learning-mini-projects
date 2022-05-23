import math
import torch

def in_circle(p1, center, radius):
    d = math.sqrt((p1[0]-center[0])**2 + (p1[1]-center[1])**2)
    return d<=radius

def generate_set(N, center, r):
    train_set = torch.empty(N, 2).uniform_()

    #0-1 labels
    train_labels = torch.empty(N)

    for i in range(len(train_set)):
        train_labels[i] = in_circle(train_set[i], center, r)

    return train_set, train_labels