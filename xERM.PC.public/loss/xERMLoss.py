import torch
import torch.nn as nn
import torch.nn.functional as F

class xERMLoss(nn.Module):
    def __init__(self, gamma=2):
        super(xERMLoss, self).__init__()
        self.XE_loss = nn.CrossEntropyLoss(reduction='none')
        self.gamma = gamma

    def forward(self, logits_F, logits_CF, logits_FCF, labels):
        # calculate w_cf
        XE_CF = self.XE_loss(logits_CF, labels)
        XE_F = self.XE_loss(logits_F, labels)

        XE_CF = torch.pow(XE_CF, self.gamma)
        XE_F = torch.pow(XE_F, self.gamma)

        w_cf = XE_CF/(XE_CF + XE_F + 1e-5)
        w_f = 1 - w_cf
        # factual loss
        loss_F = self.XE_loss(logits_FCF, labels)
        # counterfacutal loss
        prob_CF = F.softmax(logits_CF, -1).clone().detach()
        prob_FCF = F.softmax(logits_FCF, -1)
        loss_CF = - prob_CF * prob_FCF.log()
        loss_CF = loss_CF.sum(1)

        loss = (w_cf*loss_CF).mean() + (w_f*loss_F).mean()

        return loss
