import torch

print(torch.__version__)

class SimpleLayer:
    def __init__(self, in_features, out_features):
        self.weights = torch.randn(in_features, out_features, requires_grad=True)
        self.bias = torch.zeros(out_features, requires_grad=True)

    def update_W(self, w):
        self.weights = w
    
    def get_W(self):
        return self.weights
    
    def forward(self, x):
        self.x = x
        o = x @ self.weights + self.bias
        return o
    
    def backward(self, d_out):
        dW = self.x.t() @ d_out
        db = torch.sum(d_out,dim=0)

        dX = d_out @ self.weights.T

        return dX, dW, db
    

x = torch.randn(1, 10)
sl = SimpleLayer(10,1)

out = sl.forward(x)
dout = out
dX, dW, db = sl.backward(dout)

print(dout)
print(dX, dW, db)

alpha = 0.1

for i in range(101):
    if i % 20 == 0: 
        print("\nIteration", i)
        out = sl.forward(x)
        dX, dW, db = sl.backward(out)
        W_new = sl.get_W() - alpha * dW
        sl.update_W(W_new)
        print(out)
