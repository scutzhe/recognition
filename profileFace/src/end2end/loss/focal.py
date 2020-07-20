import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Support: ['FocalLoss']


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)

        # i think this is wrong
        # mean(-(1-y)^gamma*log(y)) ≠ (1-exp(mean(log(y))))^gamma * (mean(log(y)))
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class FocalLoss2(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss2, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * (1-logpt.data.exp())**self.gamma * logpt

        return loss.mean()
