import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

data = np.load("mnist.npz")
x = data["x_train"]
y = data["y_train"]

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

model = torch.load("ten-mnet.pt")
model.eval()

for value in range(30):
    xTrain = np.array(x[value+70].flatten())
    norm = np.linalg.norm(xTrain)
    xT = xTrain/norm
    xTensor = torch.tensor(xT).float()
    out = model(xTensor)
    print(out[0].item(), y[value+70])
