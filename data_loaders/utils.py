import os

import torch


def load_checkpoint(model, device, ckpt_dir, ckpt_name='best_model.pt'):
    ckpt = torch.load(os.path.join(ckpt_dir, ckpt_name), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    return model
