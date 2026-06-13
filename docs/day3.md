# Date: 6/12/2026
## Accompishments
1. Normalized the coordinates for the images because I thought the neural network would prefer them as opposed to (2,19)

## Thoughts
Last night I stayed up until 7:30am...then had work at 12:40pm...then got home at 10:30pm...but we still persist. Even if it is a few minutes everyday, I want to work on this project. The goal is to seriously have a journey entry every day until this project is completed. 

Yesterday I worked on getting the dataset inside of a dataloader and visualizing the samples. As of right now, everything is in main (though it is no longer than 20 lines of code). 

The hypernetwork will take in a image and output the weights for a INR model (of the ones we are choosing). I think today I will work on getting an INR model created. 

Before I work on the INR model (which will be a ReLU coordinate network to start off). I need to ensure that the dataloader is set and that the coordinates are normalized for the model. As the current images are all 32x32 I will change the range (from 0-31) to (0-1). 

## Things I learned
1. When I was originally doing this I was altering the color intensities. I am not sure if I did this wrong but the image I get back was completely black. I was doing min-max on the wrong values. I should have been doing them on the coordinates. Take the max coordinate (31 for CIFAR10) and the min (0) and do:
    New_x = x / 31
    
    Then I can turn this into 2 tensors that when combine will give the proper image dimensions

## Sources
Nothing was used today

## AI usage (Prompts)
1. No specific prompts today, just getting some simple coding answers. For example asking about what min max is (no code was given), asking is tensor/number would work (it does), and remembering torch.zeros_like and torch.arange