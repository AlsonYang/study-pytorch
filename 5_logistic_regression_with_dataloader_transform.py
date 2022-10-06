'''
Youtube chapter: Dataset transform
Compared to 4_*.py
- custom Transform classes `StandardScalerT` and `ToTensorT` that get fed into the Dataset
- Use standardised way to transform input data in Dataset.__call__ other than doing the processing in Dataset.__init__

Small thought:
- Although the conventional way is to apply `transform` in dataset.__getitem__. It might still be a good idea to leave some input processing in the Dataset.__init__ 
    if the processing requires seeing the full dataset to get statistic like the scikit learn `StandardScaler`. Otherwise, we will need to (which is what's been done here) read in the 
    whole dataset first, compute the stats (mean, std), and pass them into the `StandardScalerT` class, which seems less lightweight and more manual

Observations:
- n_sample = 569
- With full-patch and N_EPOCH = 200 -> Total iterations of update = 200, the training loss reduce to 0.077. 
- With mini-batch with size = 16 and N_EPOCH = 6 -> Total iteration of update (ceil(569/16) * 6) = 216, the training loss reduce to 0.080.

What it means:
    Although with mini-batch, the total iterations are higher and the training loss is higher, which might seem less preferable. 
    However, each iteration only looks at 16 samples instead of 569, so it is trained much quicker. For the same amount of training time, 
    mini-batch will achieve lower training loss. 
'''
import math
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import torchvision
from typing import Tuple
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


# seed
SEED = 42
torch.manual_seed(SEED)

# script params
PRINT_EVERY_NTH_ITERATION = 36
# PRINT_EVERY_NTH_ITERATION = 10
# model hyper-params
LEARNING_RATE = 0.01
# N_EPOCH = 200
# BATCH_SIZE = 569 # full-batch
N_EPOCH = 6
BATCH_SIZE = 16 # mini-batch

#===step 0: get data ready===
## save data to local disk for the first time
# bc = datasets.load_breast_cancer()
# X, y = bc.data, bc.target
# print(X.shape, y.shape)
# bc_all = np.concatenate([X,np.reshape(y, [y.shape[0], 1])], axis=1)
# np.savetxt('data/breast_cancer.txt', bc_all)

# get mean and std of X
bc_all = np.loadtxt('data/breast_cancer.txt')
X = bc_all[:, :-1]
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

class StandardScalerT:
    '''Input Transform class: scale input x with given scaler'''
    def __init__(self, mean, std):
        self.mean = torch.from_numpy(mean.astype(np.float32))
        self.std = torch.from_numpy(std.astype(np.float32))

    def __call__(self, sample: Tuple[Tensor, Tensor]):
        """required to be `transform`"""
        X, y = sample
        X_standardised = (X - self.mean)/self.std
        return X_standardised, y

class ToTensorT:
    '''Input Transform class: transform numpy sample into tensor sample'''
    def __call__(self, sample):
        """required to be `transform`"""
        X, y = sample
        return torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32))

class BreastCancerDataset(Dataset):
    def __init__(self, filepath, transform):
        self.transform = transform

        # load
        xy = np.loadtxt(filepath)
        self.x = xy[:, :-1]
        self.y = xy[:, [-1]]

        # metadata
        self.n_samples = xy.shape[0]
        self._in_features = self.x.shape[1]

    def __getitem__(self, index):
        """required to be `Dataset`"""
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        """required to be `Dataset`"""
        return self.n_samples

    @property
    def in_features(self):
        return self._in_features

ss = StandardScaler()
composed_transform = torchvision.transforms.Compose([ToTensorT(), StandardScalerT(X_mean, X_std)])
bc_dataset = BreastCancerDataset('data/breast_cancer.txt', transform=composed_transform)
bc_dataset[0]
bc_dataloader = DataLoader(dataset=bc_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
# print(iter(bc_dataloader).next())

#===step 1: set up model===
class LogisticRegression(nn.Module):
    def __init__(self, in_features, **kwargs):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features, 1, **kwargs)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred 

in_features = bc_dataset.in_features
n_samples = len(bc_dataset)

model = LogisticRegression(in_features=in_features)

#===step 2: set up criterion and optimizer===
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

#===step 3: training===
def get_train_setup_description(n_samples, n_epoch, batch_size):
    n_iterations = math.ceil(n_samples / batch_size)
    description = (
        f"In total, {n_samples} samples will be trained for max of {n_epoch} epochs " 
        f"with batch size {batch_size} over {n_iterations} iterations per epoch "
        f"and {n_epoch * n_iterations} iterations in total"
    )
    return description
print(get_train_setup_description(n_samples, N_EPOCH, BATCH_SIZE))

n_iterations = math.ceil(n_samples / BATCH_SIZE)
for epoch in range(N_EPOCH):
    for i, (X_train, y_train) in enumerate(bc_dataloader):
        # forward
        y_pred = model(X_train)
        l = criterion(y_pred, y_train)

        # backward
        l.backward()

        # update
        optimizer.step()
        optimizer.zero_grad()


        # info: logging 
        if (i + 1) % PRINT_EVERY_NTH_ITERATION == 0:
            (w, b) = model.parameters()
            print(f'epoch: {epoch + 1}, iteration: {i + 1}, loss: {l: .8f}, b: {b.item(): .4f}')

#===step 4: see training loss over the whole epoch===
X_train, y_train = bc_dataset[:]
with torch.no_grad():
    y_pred = model(X_train)
    l = criterion(y_pred, y_train)
    print(f"After training, l for the whole epoch is {l:.8f}")


