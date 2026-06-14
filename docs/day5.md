# Date: 6/14/2016
## Accompishments


## Thoughts


## Things I learned
Previous training loop 
'''
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
'''
Updated training loop
'''
for epoch in range (epochs):
    model.train()
    loss_tensor = torch.empty(size=(x_dim,y_dim))
    for i in range(x_dim):
        for j in range(y_dim):
            pred = model(coords[:,i,j])
            loss_tensor[i,j] = lossfn(pred, img[:,i,j])
    loss = torch.mean(loss_tensor)
    print(f"epoch: {epoch}, loss: {loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
'''
1. When you convert a tensor to a numpy array and use .detach() it removes that tensor from the computational graph used to calculate gradients. Because of this, it was never preforming backprob (which is why it wasn't training). TO fix this, I created a tensor (img sized) and updated the losses there. Then I just called torch.mean(loss_tensor) to get a single valued and loss.backwards() was used on that single value. 
        

## Sources


## AI usage (Prompts)
