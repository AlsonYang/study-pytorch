'''
ref: [Deep Learning With PyTorch - Full Course by Python Engineer]<https://www.youtube.com/watch?v=c36lUUr864M> 1:27:14 to 1:39:30

Implement Linear regression using torch building blocks and numpy's data
'''

import numpy as np
import torch
from sklearn import datasets
import matplotlib.pyplot as plt

#===training params===
LEARNING_RATE=1e-2
N_EPOCH = 300
PRINT_EVERY_NTH = 2
#===create data===
x_np, y_np, coef = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4, bias=5, coef=True)
x = torch.from_numpy(x_np.astype(np.float32))
y = torch.from_numpy(y_np.astype(np.float32))
y = y.view(x.shape[0], 1)

#===setup===
class LinearRegression(torch.nn.Module):
    '''5.b) This is a dummy class that wraps the torch.nn.Linear model
        The purpose is to demonstrate how to create a bare-minimum custom model
    '''
    def __init__(self, input_dim, output_dim, **kwargs) -> None:
        super(LinearRegression, self).__init__()
        self.lin = torch.nn.Linear(input_dim, output_dim, **kwargs)
    
    def forward(self, x):
        return self.lin(x)

# model = torch.nn.Linear(x.shape[1], y.shape[1], bias=True)
model = LinearRegression(x.shape[1], y.shape[1], bias=True)
loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

#===training===
for epoch in range(1, N_EPOCH + 1):
    # forward
    y_pred = model(x)

    # loss 
    l = loss(y_pred, y)

    # update
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

    # x) logging 
    if epoch % PRINT_EVERY_NTH == 0:
        (w, b) = model.parameters()
        print(l)
        print(type(l))
        print(f'epoch: {epoch}, w: {w.item():.3f}, b: {b.item():.3f} loss: {l: .8f}')

#===viz after training===
print([model.parameters()])
predicted = model(x).detach().numpy()
plt.plot(x_np, y_np, 'ro')
plt.plot(x_np, predicted, 'b')
plt.show()