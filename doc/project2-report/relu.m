class ReLU(Module):
    def __init__(self):
        self.input = None
    
    def relu(self, x):
        z = empty(x.size()).fill_(0)
        relu = x.maximum(z)
        return relu
    
    def d_relu(self, x):
        c = x.detach().clone()
        d_relu = x.apply_(lambda x: 1 if x > 0 else 0)
        return d_relu
        
    def forward (self, *input):
        self.input = input
        return tuple([self.relu(tensor) for tensor in input])
        
    def backward (self, *gradwrtoutput):
        backwards_relu = tuple([gradwrtoutput[i] * self.d_relu(self.input[i]) for i in range(len(self.input))])
        return backwards_relu