import torch
import torch.nn as nn
from utils import *
from os import path
import math


class BaselineCosineClassifier(nn.Module):
    def __init__(self, num_classes=10, feat_dim=128, tau=16, gamma=0.03125):
        super(BaselineCosineClassifier, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim).cuda(), requires_grad=True)
        self.tau = tau
        self.gamma = gamma
        self.reset_parameters(self.weight)

    def reset_parameters(self, weight):
        stdv = 1. /math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)

    def get_cos_sin(self, x, y):
        cos_val = (x * y).sum(-1, keepdim=True) / torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
        sin_val = (1 - cos_val * cos_val).sqrt()
        return cos_val, sin_val

    def l2_norm(self, x):
        normed_x = x / torch.norm(x, 2, 1, keepdim=True)
        return normed_x

    def causal_norm(self, x, weight):
        norm= torch.norm(x, 2, 1, keepdim=True)
        normed_x = x / (norm + weight)
        return normed_x

    def forward(self, x, label, embed):
        normed_w = self.causal_norm(self.weight, self.gamma)
        normed_x = self.l2_norm(x)
        y = torch.mm(normed_x * self.tau, normed_w.t())
        
        return y, None

    
def create_model(feat_dim, num_classes=1000, stage1_weights=False, dataset=None, log_dir=None, test=False, use_effect=True, num_head=None, tau=None, alpha=None, gamma=None, *args):
    print('Loading Baseline Cosine Classifier')
    clf = BaselineCosineClassifier(num_classes, feat_dim, tau=tau, gamma=gamma)

    return clf