class GD(object):
    "Gradient descent for MSE loss"
    def __init__(self, params, lr):
        if not isinstance(params, list):
            params = [params]
        self.params = params
        self.lr = lr
    
    def update(self, param):
        for i in range(len(param['weight'])):        
            for j in range(len(param['grad'][0])):
                grad = param['grad'][i][j]
            param['weight'][i] -= self.lr * grad / len(param['grad'][0])
    
    def step(self):
        for param in self.params:
            self.update(param)
            
    def zero_grad(self):
        for param in self.params:
            for i in range(len(param['weight'])):
                param['grad'][i].clear() # (1) this can be optimized