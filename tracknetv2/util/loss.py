import torch


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, input):
        a = (input ** 2) * (1 - target) * torch.clamp(1 - input, 1e-7, 1).log()
        b = ((1 - input) ** 2) * target * torch.clamp(input, 1e-7, 1).log()
        loss = a - b
        return loss