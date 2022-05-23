class Sequential(Module):
    def __init__(self, modules):
        self.modules = modules
        
    def forward(self, *input):
        x = input
        n = len(self.modules)
        for i in range(n):
            x = self.modules[i].forward(*x)
        return x
        
    def backward(self, *gradwrtoutput):
        x = gradwrtoutput
        n = len(self.modules)
        for i in range(n):
            x = self.modules[n-i-1].backward(*x)
        return x
        
    def param(self):
        return [module.param() for  module in self.modules if module.param() != []]