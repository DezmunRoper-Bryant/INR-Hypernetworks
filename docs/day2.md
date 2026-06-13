# Date: 6/11/2026
## Accompishments

1. Review basic INRs and what they actually are
2. Used torchvision.datasets.CIFAR10 and torch.util.data.DataLoader to download and store the dataset into a dataloader that is put inside of a iterator and iterated through (with next()) to print out the dataset

## Thoughts
This is project 1 for my June-December grind. Project 2 was the recreation of various GNN papers but funny enough I will have to attempt to recreate the INRs from research papers as well. A LOT of papers will be read over the next few months sadly. 

The purpose of this project is the design a hypernetwork capable of taking images and giving the weights for a INR model. There are two challenges for this project (aside from coding difficulties)
1. I would like to recreate 3-5 different INR models and compare these. This is require that I read some papers and design the models from scratch.
2. Will model will require a different Hypernetwork? Or can I get away with using the same hypernet regardless of the INR I use? I would imagine the hypernet would need to change because from my memory the FFN is more complicated than the SIREN so the SIREN wouldn't need as complicated of a Hypernet. I'll figure this part out eventually.  

By the way...since I am using a Dell XPS 15 I will only be evaluating the CIFAR10 dataset. I'll move to much higher res images once I get a more suitable PC. 

Maybe we can look at the 1st and 2nd derivative of the recreated images and ALSO compare those with the original? I saw this in a SIREN video:

## Things I learned
1. A INR encodes a SINGLE image into a nueral network that operates like a look up table. You specify a coordinate and the model returns the color value at that coordinate. The standard baseline for INRs are overfit so a single network cooresponds with a single image. BUT there is another type of INR called the Conditioned INR that takes a pixel location AS WELL AS a latent code for the image. But I have learned that the hypernetwork I am creating is better computationally than the conditional INR (it is just more complex and expensive to train)
2. When I am trying to use a folder, I need "./folder" rather than "/folder" because "/" starts at system/root while "./" starts where I am running the python code at
3. The torchvision.datasets.CIFAR10 images are PIL (pillow)
4. YOU CAN NOT USE np.reshape to go from 3,32,32 to 32,32,3. It will scramble the pixels. Instead you need to use np.transpose(img, (new dim order)) with 0 being the first dimension. So I did np.transpose(img, (1,2,0))
5. Orginally the next(iter(dataloader)) was not working. This is because it does not work with PIL images. The work around for this was to transform the PIL images into tensors using transform = torchvision.transforms.PILToTensor inside of the torchvision.datasets.CIFAR10 class definition


## Sources
1. https://www.youtube.com/watch?v=Q2fLWGBeaiI - Used to get an idea of INRs
2. https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html
3. https://docs.pytorch.org/docs/2.12/data.html#torch.utils.data.DataLoader
4. https://www.cs.toronto.edu/~kriz/cifar.html
5. https://discuss.pytorch.org/t/getting-typeerror-default-collate-batch-must-contain-tensors-numpy-arrays-numbers-dicts-or-lists-found-class-pil-image-image/161703 - Used to troubleshoot an issue where iter was giving 
    "
    TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>
    "

## AI usage (Prompts)
1. Day 2: The goal is to get the CIFAR10 dataset up and loaded onto my computer to start. Issue. I have NO CLUE how to do that nor do I want you to help. How should I go about figuring out this problem
2. how do I call file locations in my project folder? "/data" goes all the way into  my local disk when I wanted it to call the /data folder in my project
3. should the data folder be in src?
    YES I KNOW THIS IS A DUMB QUESTION
4. Untracked files:
    (use "git add <file>..." to include in what will be committed)
        .gitignore
        data/
        docs/day2.md
        src/

    if I do git add . 
    and the gitignore ignored the data folder, will something bad happen?
