'''
Compare to 2_*.py
- Linear regression -> Logistic regression
- StandardScaler for input featuers
'''

import torch 
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# script params
PRINT_EVERY_NTH = 10
# model hyper-params
LEARNING_RATE = 0.01
N_EPOCH = 200

#===step 0: get data ready===
# get data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
print(X.shape, y.shape)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# scale: without scaling, the model doesn't learn at all, probably because the value Z are too big given X, and so gradient of activation is 0 # Important!!!
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# convert to torch.tensor
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
print(X_train.shape, y_train.shape)

# reshape y for NN compatibility
y_train = y_train.view([y_train.shape[0], 1])
y_test = y_test.view([y_test.shape[0], 1])
print(X_train.shape, y_train.shape)

#===step 1: set up model===
class LogisticRegression(nn.Module):
    def __init__(self, in_features, **kwargs):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features, 1, **kwargs)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred 

in_features = X_train.shape[1]
model = LogisticRegression(in_features=in_features)

#===step 2: set up criterion and optimizer===
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

#===step 3: training===
for epoch in range(N_EPOCH):
    # forward
    y_pred = model(X_train)
    l = criterion(y_pred, y_train)

    # backward
    l.backward()

    # update
    optimizer.step()
    optimizer.zero_grad()

    # logging 
    if (epoch + 1) % PRINT_EVERY_NTH == 0:
        (w, b) = model.parameters()
        print(f'epoch: {epoch + 1}, loss: {l: .8f}, b: {b.item(): .4f}')

#===step 4: eval===
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_cls = y_pred.round()
    acc = y_pred_cls.eq(y_test).sum() / y_test.shape[0]
    print(f'accuracy: {acc.item():.4f}')
