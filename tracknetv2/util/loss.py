import torch


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, input, reduction='sum'):
        a = (input ** 2) * (1 - target) * torch.clamp(1 - input, 1e-10, 1).log()
        b = ((1 - input) ** 2) * target * torch.clamp(input, 1e-10, 1).log()
        loss = -1 * (a + b)
        return loss.sum() if reduction == 'sum' else loss.mean()