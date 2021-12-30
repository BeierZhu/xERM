import torch
import torch.nn as nn
import torch.nn.functional as F

class xERMLoss(nn.Module):
    def __init__(self, gamma):
        super(xERMLoss, self).__init__()
        self.CE_loss = nn.CrossEntropyLoss(reduction='none')
        self.gamma = gamma

    def forward(self, logits_TE, logits_TDE, logits_student, labels):
        # calculate weight
        TDE_acc = self.CE_loss(logits_TDE, labels)
        TE_acc = self.CE_loss(logits_TE, labels)
        TDE_acc = torch.pow(TDE_acc, self.gamma)
        TE_acc = torch.pow(TE_acc, self.gamma)
        weight = TDE_acc/(TDE_acc + TE_acc)
        # student td loss
        te_loss = self.CE_loss(logits_student, labels)

        # student tde loss
        prob_tde = F.softmax(logits_TDE, -1).clone().detach()
        prob_student = F.softmax(logits_student, -1)
        tde_loss = - prob_tde * prob_student.log()
        tde_loss = tde_loss.sum(1)

        loss = (weight*tde_loss).mean() + ((1 - weight)*te_loss).mean()

        return loss
