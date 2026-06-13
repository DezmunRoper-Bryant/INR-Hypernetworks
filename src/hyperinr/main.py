import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np

'''
Implementing Min-Max Normalization
x_new = (x - x_min) / (x_max - x_min)
'''

# def normalize(img:torch.tensor) -> torch.tensor:
#     '''
#     This function implements min-max normalization for a single rgb image
#     Returns two tensors that when combined give every coordinate for the image
#     '''
#     x_range = img.shape[1] - 1
#     y_range = img.shape[2] - 1
#     x_tensor = torch.arange(start=0, end=img.shape[1], step=1) / x_range
#     y_tensor = torch.arange(start=0, end=img.shape[2], step=1) / y_range
#     return x_tensor, y_tensor

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
    coords = normalize(img)
    print(f"x: {coords[0]}")
    print(f"y: {coords[1]}")
    print(coords[:,0,31])
    img = np.transpose(img, (1,2,0)) # transpose is needed to show the img
    plt.imshow(img)
    plt.show()


