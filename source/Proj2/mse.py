from torch import empty
from module import *
    
class MSE(Module):
    def __init__(self):
        self.input = None

    def mse(self, pred, target):
        mse = ((pred - target)**2).mean()
        
        return mse

    def d_mse(self, pred, target):
        dmse = 2*((pred - target)).mean()
        
        return dmse
    
    def forward(self, *input):
        if len(input) % 2 != 0:
            raise ValueError('Error: the input must be evenly sized')
        
        middle_index = len(input)//2
        self.pred = tuple(list(input)[: middle_index])
        self.labels = tuple(list(input)[middle_index:])
        
        return tuple([self.mse(self.pred[i], self.labels[i])] for i in range(len(self.pred)))

    def backward(self):
        return tuple([self.d_mse(self.pred[i], self.labels[i]).unsqueeze(0) for i in range(len(self.pred))])