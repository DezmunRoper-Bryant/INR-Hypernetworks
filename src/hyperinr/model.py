'''
This file will be use for the baseline INR model
Input: (x, y) coordinate
Output: (r,g,b) pixel
'''

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import numpy as np


class INRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = 10
        self.linear1 = nn.Linear(in_features=2, out_features=self.hidden)
        self.linear2 = nn.Linear(in_features=self.hidden, out_features=3)
        self.relu = nn.ReLU()
    
    def forward(self, x:torch.tensor):
        x = self.linear1(x)
        x = self.relu(x)
        y = self.linear2(x)
        return y
    
def normalize(img:torch.tensor) -> torch.tensor:
    '''
    This function implements min-max normalization for a single rgb image
    Returns a coordinate tensor
    '''
    x_range = img.shape[1]
    y_range = img.shape[2]

    coord_img = torch.empty(2, x_range, y_range)
    print(coord_img.shape)

    x_ = torch.arange(start=0, end=x_range, step=1) / (x_range - 1)
    for i in range(x_range):
        coord_img[0,i] = x_

    y_ = torch.arange(start=0, end=y_range, step=1) / (y_range - 1)
    for j in range(y_range):
        coord_img[1,:,j] = y_
        
    return coord_img

batches = 1

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform = torchvision.transforms.PILToTensor())
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batches, shuffle = True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batches, shuffle = True)

loader = iter(train_dataloader)
img, label = next(loader)
img = img[0]
coords = normalize(img)
_, x_dim, y_dim = coords.shape


epochs = 3
model = INRModel()

lossfn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


for epoch in range(epochs):
    model.train()
    loss_store = np.array([])
    for i in range(x_dim):
        for j in range(y_dim):
            pred = model(coords[:,i,j])
            logit = lossfn(pred, img[:,i,j])
            logit = logit.detach().numpy()
            loss_store = np.append(loss_store, logit)
    loss_np = np.mean(loss_store)
    loss = torch.tensor(loss_np, requires_grad=True)
    print(f"epoch: {epoch}, loss: {loss_np}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

