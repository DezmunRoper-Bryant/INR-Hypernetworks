import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np

batches = 64

## get the dataset inside of the dataloaders
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform = torchvision.transforms.PILToTensor())
print(train_dataset)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batches, shuffle = True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
print(test_dataset)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batches, shuffle = True)

loader = iter(train_dataloader)
for i in range(batches):
    img, label = next(loader)
    img = np.transpose(img[0], (1,2,0))
    plt.imshow(img)
    plt.show()