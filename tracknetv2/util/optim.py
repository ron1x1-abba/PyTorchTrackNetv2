import torch


OPTIMIZERS = {
    'adadelta': torch.optim.Adadelta,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'sgd': torch.optim.SGD
}
