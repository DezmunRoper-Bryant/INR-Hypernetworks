import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np

'''
Implementing Min-Max Normalization
x_new = (x - x_min) / (x_max - x_min)
'''

def normalize(img:torch.tensor) -> torch.tensor:
    '''
    This function implements min-max normalization for a single rgb image
    '''
    x_range = img.shape[1] - 1
    y_range = img.shape[2] - 1
    x_tensor = torch.arange(start=0, end=img.shape[1], step=1) / x_range
    y_tensor = torch.arange(start=0, end=img.shape[2], step=1) / y_range

    return x_tensor, y_tensor

batches = 64

## get the dataset inside of the dataloaders
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform = torchvision.transforms.PILToTensor())
# print(train_dataset)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batches, shuffle = True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
# print(test_dataset)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batches, shuffle = True)

loader = iter(train_dataloader)
for i in range(batches):
    img, label = next(loader)
    img = img[0]
    x , y = normalize(img)
    print(f"x: {x}")
    print(f"y: {y}")
    img = np.transpose(img, (1,2,0))
    plt.imshow(img)
    plt.show()

