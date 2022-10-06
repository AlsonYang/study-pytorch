"""
- MNIST
- DataLoader, Transformation
- Multilayer Neural Net with activation functions
- Loss and optimizer
- Training loop
- Model evaluation
- GPU support? -> Need special version for M1, need to investigate
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# seed
SEED = 42
torch.manual_seed(SEED)

#
PRINT_EVERY_NTH_ITERATION = 100

LEARNING_RATE = 1e-3
BATCH_SIZE = 128
N_EPOCH = 3

# regularization
L2_NORM = 0

#===data====
# # inspection on the raw data
# X0 = torchvision.datasets.MNIST('data', train=False, transform=ToTensor())[0][0]
# y0 = torchvision.datasets.MNIST('data', train=False, transform=ToTensor())[0][1]
# print(X0.shape) # [1, 28, 28]: 1 channel, 28*28 pixel

# plt.imshow(X0[0], cmap='gray') # only get the value of the first channel, as it only has one channel anyway
# plt.show()
# print(y0)


# MNIST Dataloader and Transformation
class Flatten:
    def __call__(self, X):
        return X.reshape(-1) # (1, 28, 28) -> (784,)

mnist = torchvision.datasets.MNIST('data', train=True, transform=Compose([ToTensor(), Flatten()]))
dataloader = DataLoader(dataset=mnist, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
n_samples = len(mnist) #60000
n_features = mnist[0][0].shape[0]
n_classes = 10



# model
class MultiLayer(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.linear1 = nn.Linear(n_features, 256)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(64, n_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.out(out)
        return out

# Loss and optimizer
model = MultiLayer(n_features=n_features, n_classes=n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_NORM)
# Training loop
for epoch in range(N_EPOCH):
    for i, (X, y) in enumerate(dataloader):
        # forward
        y_pred = model(X)
        loss = criterion(y_pred, y)

        # backward
        loss.backward()

        # update
        optimizer.step()
        optimizer.zero_grad()

        # info: logging 
        if (i + 1) % PRINT_EVERY_NTH_ITERATION == 0:
            print(f'epoch: {epoch + 1}, iteration: {i + 1}, loss: {loss: .8f}')


# ==== evaluation=====
def evaluate(dataset) -> float:
    with torch.no_grad():
        data = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False, drop_last=False)
        X, y = iter(data).next()
        # print(f"X.shape, y.shape: {X.shape, y.shape}")
        
        y_pred = model(X)
        y_cls = torch.argmax(y_pred, dim=1)
        micro_accuracy = y_cls.eq(y).sum()/y.shape[0]
        # print(f"y_cls.shape, y.shape: {y_cls.shape, y.shape}")
        print(f"The micro accuracy is: {micro_accuracy}")
        return micro_accuracy

# Model evaluation on train
train_dataset = torchvision.datasets.MNIST('data', train=True, transform=Compose([ToTensor(), Flatten()]))
train_accuracy = evaluate(train_dataset)
# Model evaluation on test
test_dataset = torchvision.datasets.MNIST('data', train=False, transform=Compose([ToTensor(), Flatten()]))
test_accuracy = evaluate(test_dataset)

    