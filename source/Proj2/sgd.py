class SGD(object):
    def __init__(self, params, lr):
        if not isinstance(params, list):
            params = [params]
        self.params = params
        self.lr = lr
    
    def update(self, param):
        for i in range(len(param['weight'])):
            for j in range(len(param['grad'][0])):
                param['weight'][i] -= self.lr * param['grad'][i][j]
    
    def step(self):
        for param in self.params:
            self.update(param)
            
    def zero_grad(self):
        for param in self.params:
            for i in range(len(param['weight'])):
                param['grad'][i].clear() # (1) this can be optimized