'''
ref: [Deep Learning With PyTorch - Full Course by Python Engineer]<https://www.youtube.com/watch?v=c36lUUr864M> 0:0 to 1:27:14
 
Implement of linear regression from most manual into higher level 
1. numpy array
2. pytorch tensor
3. pytorch tensor with torch autograd
4. pytorch tensor with torch autograd loss optimizer
5.a pytorch tensor with torch autograd loss optimizer model
5.b pytorch tensor with torch autograd loss optimizer custom_model
'''
from pickletools import optimize
import numpy as np
import torch
#====train params=====
N_EPOCH = 71
LEARNING_RATE = 1e-2
PRINT_EVERY_NTH = 2

#===LR functions====
def forward(x, w):
    '''1,2,3,4) numpy and torch'''
    return x * w

def loss(y, y_pred):
    '''1,2,3) numpy and torch'''
    return ((y_pred - y)**2).mean()

def gradient(x, y_pred, y):
    '''1,2) numpy and torch'''
    return (2*x*(y_pred - y)).mean()

def train(x, y, w):
    '''1, 2) numpy and torch'''
    for epoch in range(N_EPOCH):
        # 1) forward
        y_pred = forward(x, w)
        # 2) compute loss
        l = loss(y, y_pred)
        # 3) calculate gradients and update param
        dw = gradient(x, y_pred, y)
        w -= LEARNING_RATE * dw
        # x) logging
        if epoch % PRINT_EVERY_NTH == 0:
            print(f'epoch: {epoch}, w: {w:.3f}, loss: {l: .8f}')
    return w

def train_with_auto_grad(x, y, w):
    '''3) torch only'''
    for epoch in range(N_EPOCH):
        # 1) forward
        y_pred = forward(x, w)
        # 2) compute loss
        l = loss(y, y_pred)

        # 3) calculate gradients and update param
        l.backward()

        # prefered: keep using existing w but need to zero out w.grad after each iteration
        with torch.no_grad(): # temporarily disable grad graph
            w -= LEARNING_RATE * w.grad    
        w.grad.zero_() # clear grad accumulator

        # # less prefered: create new w, but need to reset required_grad to True
        # with torch.no_grad(): # temporarily disable grad graph
        #     w = w - LEARNING_RATE * w.grad
        # w.requires_grad_(True)
        
        # x) logging
        if epoch % PRINT_EVERY_NTH == 0:
            print(f'epoch: {epoch}, w: {w:.3f}, loss: {l: .8f}')
    return w



def train_with_auto_grad_loss_optimizer(x, y, w):
    '''4) torch only'''
    t_loss = torch.nn.MSELoss()
    t_optimizer = torch.optim.SGD([w], lr=LEARNING_RATE)
    for epoch in range(N_EPOCH):
        # 1) forward
        y_pred = forward(x, w)

        # 2) compute loss
        l = t_loss(y, y_pred)
        
        # 3) calculate gradients and update param
        l.backward()
        t_optimizer.step()
        t_optimizer.zero_grad() # clear grad accumulator
        
        # x) logging
        if epoch % PRINT_EVERY_NTH == 0:
            print(f'epoch: {epoch}, w: {w:.3f}, loss: {l: .8f}')
    return w

def train_with_auto_grad_loss_optimizer_model(x, y):
    '''5.a) torch only'''
    t_model = torch.nn.Linear(x.shape[1], y.shape[1], bias=False)
    t_loss = torch.nn.MSELoss()
    t_optimizer = torch.optim.SGD(t_model.parameters(), lr=LEARNING_RATE)

    for epoch in range(N_EPOCH):
        # 1) forward
        y_pred = t_model(x)

        # 2) compute loss
        l = t_loss(y, y_pred)
        
        # 3) calculate gradients and update param
        l.backward()
        t_optimizer.step()
        t_optimizer.zero_grad() # clear grad accumulator
        
        # x) logging
        if epoch % PRINT_EVERY_NTH == 0:
            (w, ) = t_model.parameters()
            print(f'epoch: {epoch}, w: {w.item():.3f}, loss: {l: .8f}')
    return w

class LinearRegression(torch.nn.Module):
    '''5.b) This is a dummy class that wraps the torch.nn.Linear model
        The purpose is to demonstrate how to create a bare-minimum custom model
    '''
    def __init__(self, input_dim, output_dim, **kwargs) -> None:
        super(LinearRegression, self).__init__()
        self.lin = torch.nn.Linear(input_dim, output_dim, **kwargs)
    
    def forward(self, x):
        return self.lin(x)

def train_with_auto_grad_loss_optimizer_custom_model(x, y):
    '''5.b) torch only
    '''
    t_model = LinearRegression(x.shape[1], y.shape[1], bias=False)
    t_loss = torch.nn.MSELoss()
    t_optimizer = torch.optim.SGD(t_model.parameters(), lr=LEARNING_RATE)

    for epoch in range(N_EPOCH):
        # 1) forward
        y_pred = t_model(x)

        # 2) compute loss
        l = t_loss(y, y_pred)
        
        # 3) calculate gradients and update param
        l.backward()
        t_optimizer.step()
        t_optimizer.zero_grad() # clear grad accumulator
        
        # x) logging 
        if epoch % PRINT_EVERY_NTH == 0:
            (w, ) = t_model.parameters()
            print(f'epoch: {epoch}, w: {w.item():.3f}, loss: {l: .8f}')
    return w

#====1. numpy derived grad=====
print('=== 1) numpy derived grad ===')
x = np.array([1,2,3,4], dtype=np.float32)
y = x * 2
w = 0.0
print(x, y, w)

trained_w = train(x, y, w)
print(trained_w) # 1.9999805045127872

#====2. torch derived grad=====
print('=== 2) torch derived grad ===')
x = torch.tensor([1,2,3,4], dtype=torch.float32)
y = x * 2
w = 0.0
print(x, y, w)

trained_w = train(x, y, w)
print(trained_w.item()) # 1.9999804496765137

#====3. torch autograd=====
print('=== 3) torch auto grad ===')
x = torch.tensor([1,2,3,4], dtype=torch.float32)
y = x * 2
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
print(x, y, w)

trained_w = train_with_auto_grad(x, y, w)
print(trained_w.item()) # 1.9999804496765137


#====4. torch autograd + t_loss + t_optimizer
print('=== 4) torch autograd + autoloss + autooptimizer ===')
x = torch.tensor([1,2,3,4], dtype=torch.float32)
y = x * 2
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
print(x, y, w)


trained_w = train_with_auto_grad_loss_optimizer(x, y, w)
print(trained_w.item()) # 1.9999804496765137

#====5. torch autograd + t_loss + t_optimizer + t_model
print('=== 5.a) torch autograd + autoloss + autooptimizer + t_model ===')
x = torch.tensor([1,2,3,4], dtype=torch.float32)
x = x.view(x.shape[0], 1) # the torch.nn model requires x and y to be of shape [n_sample, n_features]
y = x * 2
print(x, y)

trained_w = train_with_auto_grad_loss_optimizer_model(x, y)
print(trained_w.item()) # ~1.99997878074646 This is different because the nn.Linear initialize w differently. It is also different everytime

print('=== 5.b) torch autograd + autoloss + autooptimizer + custom_model ===')
trained_w2 = train_with_auto_grad_loss_optimizer_custom_model(x, y)
print(trained_w2.item())




# from torch import nn
# a = nn.MSELoss()
# print(help(a))
# print(a)
# print(type(a))