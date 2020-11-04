import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def numerical_derivative(f, x):
    delta_x = 1e-4
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)
        
        x[idx] = tmp_val - delta_x
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2*delta_x)
        
        x[idx] = tmp_val
        it.iternext()
        
    return grad

class LogicGate:
    
    def __init__(self, gate_name, xdata, tdata):
        
        self.name = gate_name
        
        self.__xdata = xdata.reshape(4, 2)
        self.__tdata = tdata.reshape(4, 1)
        
        self.__W = np.random.rand(2, 1)
        self.__b = np.random.rand(1)
        
        self.__learning_rate = 1e-2
        
    def __loss_func(self):
        
        delta = 1e-7 # prevent log infinite
        
        z = np.dot(self.__xdata, self.__W) + self.__b
        y = sigmoid(z)
        
        # cross-entropy
        return -np.sum( self.__tdata * np.log(y + delta) + (1 - self.__tdata)*np.log((1 - y)+ delta))
    
    def error_val(self):
        
        delta = 1e-7 # prevent log infinite
        
        z = np.dot(self.__xdata, self.__W) + self.__b
        y = sigmoid(z)
        
        # cross-entropy
        return -np.sum( self.__tdata * np.log(y + delta) + (1 - self.__tdata)*np.log((1 - y)+ delta))
    
    def train(self):
        
        f = lambda x : self.__loss_func()
        
        print('Initial error value = ', self.error_val())
        
        for step in range(8001):
            
            self.__W -= self.__learning_rate * numerical_derivative(f, self.__W)
            self.__b -= self.__learning_rate * numerical_derivative(f, self.__b)
            
            if (step % 400 == 0):
                print('step = ', step, 'error value = ', self.error_val())
        
    def predict(self, input_data):
        
        z = np.dot(input_data, self.__W) + self.__b
        y = sigmoid(z)
        result = None
        
        if y > 0.5:
            result = 1
        else:
            result = 0
        return y, result
        