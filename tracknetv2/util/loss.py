import torch


class CustomLoss(torch.nn.Module):
    def __init__(self, reduction):
        super().__init__()

        self.reduction = reduction

    def forward(self, target, input):
        a = (input ** 2) * (1 - target) * torch.clamp(1 - input, 1e-10, 1).log()
        b = ((1 - input) ** 2) * target * torch.clamp(input, 1e-10, 1).log()
        loss = -1 * (a + b)
        return loss.sum() if self.reduction == 'sum' else loss.mean()