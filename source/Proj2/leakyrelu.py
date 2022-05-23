from torch import empty
from module import *

class LeakyReLU(Module):
    
    def __init__(self, par):
        self.input = None
        self.par = par
    
    def relu(self, x):
        relu = x.apply_(lambda x: x if x>0 else self.par*x)
        return relu
    
    def d_relu(self, x):
        c = x.detach().clone()
        d_relu = x.apply_(lambda x: 1 if x > 0 else self.par)
        return d_relu
        
    def forward (self, *input):
        self.input = input

        return tuple([self.relu(tensor) for tensor in input])
        
    def backward (self, *gradwrtoutput):
        backwards_relu = tuple([gradwrtoutput[i] * self.d_relu(self.input[i]) for i in range(len(self.input))])
        return backwards_relu