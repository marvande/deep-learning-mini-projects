from torch import empty
from module import *

class Linear(Module):
    
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.params = {}
        self.w = empty(out_features, in_features).uniform_()
        self.params['weight'] = [self.w]
        self.gradwrt_w = []
        self.params['grad'] = [self.gradwrt_w]
        if bias:
            self.b = empty(out_features).uniform_()
            self.params['weight'].append(self.b)
            self.gradwrt_b = []
            self.params['grad'].append(self.gradwrt_b)
        self.input = None
        
    def forward(self, *input):
        self.input = input
        l = []
        
        if self.bias:
            l = [self.w @ tensor + self.b for tensor in input]
        else:
            l = [self.w @ tensor for tensor in input]
        
        return tuple(l)
        
    def backward(self, *gradwrtoutput):
        l = []
        for i in range(len(gradwrtoutput)):
            # with respect to the input
            l += [(gradwrtoutput[i] @ (self.w))]
            
            # with respect to the weight
            gradwrt_w = gradwrtoutput[i].view(-1,1).mm(self.input[i].view(1,-1))
            if len(self.gradwrt_w) != len(gradwrtoutput):
                self.gradwrt_w.append(empty(gradwrt_w.size()).fill_(0).squeeze()) # (1) this can be optimized
            self.gradwrt_w[i].add_(gradwrt_w.squeeze())
         
            # with respect to the bias
            if self.bias:
                gradwrt_b = gradwrtoutput[i]
                if len(self.gradwrt_b) != len(gradwrtoutput):
                    self.gradwrt_b.append(empty(gradwrt_b.size()).fill_(0).squeeze()) # (1) this can be optimized
                self.gradwrt_b[i].add_(gradwrt_b.squeeze())        
        return tuple(l)
        
    def param(self):
        return self.params