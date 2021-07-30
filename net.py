from tqdm import trange
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

data = np.load("mnist.npz")
x = data["x_train"]
y = data["y_train"]


# 28 x 28
# 784 len

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 1)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

model = Net()

lossF = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(5):
    print("epoch: ", epoch)
    for value in trange(len(x)):
        xTrain = np.array(x[value].flatten())
        norm = np.linalg.norm(xTrain)
        xT = xTrain/norm
        xTensor = torch.tensor(xT).float()
        yTensor = torch.tensor(np.array(y[value])).float()

        output = model(xTensor)

        optimizer.zero_grad()
        
        loss = lossF(output, yTensor.unsqueeze(-1))
        loss.backward()
        optimizer.step()
    print("done")

torch.save(model, "ten-mnet.pt")
