import torch

def save_checkpoint(epoch, model, optimizer,loss,path):
    """
    Save the model check point
    inputs:
        epoch (int): number of epoch
        model(torch.model): model
        optimizer(torch.optim): torch optimizer
        loss(tensor) : loss function value
        path (str) : path to save the model
    """
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)
    