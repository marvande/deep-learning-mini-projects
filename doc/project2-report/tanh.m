class Tanh(Module):
    def __init__(self):
        self.input = None
        
    def tanh(self, x):
        ex = x.exp()
        emx = (-x).exp()
        return (ex - emx)/(ex + emx)
    
    def d_tanh(self, x):
        ex = x.exp()
        emx = (-x).exp()
        return 4/(ex + emx).pow(2)
        
    def forward (self, *input):
        self.input = input
        th = tuple([self.tanh(t) for t in input])
        return th
        
    def backward (self,*gradwrtoutput):   
        return tuple([gradwrtoutput[i]*self.d_tanh(self.input[i]) for i in range(len(self.input))])