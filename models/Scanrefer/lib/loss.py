import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxRankingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # input check
        assert inputs.shape == targets.shape
        
        # compute the probabilities
        probs = F.softmax(inputs + 1e-8, dim=1)

        # reduction
        loss = -torch.sum(torch.log(probs + 1e-8) * targets, dim=1).mean()

        return loss

class SigmoidRankingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # input check
        assert inputs.shape == targets.shape
        
        # compute the probabilities
        probs = torch.sigmoid(inputs)

        # reduction
        loss = -torch.sum(torch.log(probs + 1e-8) * targets, dim=1).mean()

        return loss