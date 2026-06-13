# Date: 6/13/2026
## Accompishments
1. Remade the normalization function
2. Constructed a simple INR model with a relu activation function in the middle

## Thoughts
Part 1 (Morning)
I changed the normalization function to create a grid (2,32,32) rather than two (32,32) for x and y. This was so that calling both the coordinates and rgb are consistent in the training loop

I make the simple INR model. 2 input, 3 output, a single layer in between

I tried to create a training loop as well but it is not training. The code takes a single image from the dataset and for every epoch it goes through every pixel and guesses the color and calculates the loss. THEN coverts that loss tensor to a numpy int and appends that to the loss_numpy and the calulates the mean of that numpy array THEN converts that back to a tensor. Yeah so that isn't training...

As I typed this out, I could have just makes a 32x32 tensor and just appened the loss values then calculated the mean of that entire tensor. But I have work so I when do that at 10:00 when I get home

Part 2 (Night)


## Things I learned
1. torch.nn.Linear already has a bias included, y = Wx + b. For some reason I thought it never had a bias, or rather I never thought about it deeply
2. Epoch in range, model.train, calc loss, optimizer zero grad, loss back, optimizer step

## Sources
1. https://discuss.pytorch.org/t/about-the-nn-module-forward/20858
2. https://arxiv.org/abs/2604.15047 Read a little bit on different INRs. There are a ton here
3. https://docs.pytorch.org/docs/2.12/generated/torch.nn.Linear.html
4. https://medium.com/data-science/the-unofficial-pytorch-optimization-loop-song-89657dd3a434

## AI usage (Prompts)
