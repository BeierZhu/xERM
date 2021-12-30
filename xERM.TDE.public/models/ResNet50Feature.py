"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


from models.ResNetFeature import *
from utils import *
from os import path
from collections import OrderedDict


def load_model(model, pretrain):
    print("Loading Backbone pretrain model from {}......".format(pretrain))
    model_dict = model.state_dict()
    pretrain_dict = torch.load(pretrain)
    new_dict = OrderedDict()

    for k, v in pretrain_dict.items():
        if k.startswith("module"):
            k = k[7:]
        if "fc" not in k and "classifier" not in k:
            new_dict[k] = v

    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    print("Backbone model has been loaded......")

    return model
        
def create_model(use_fc=False, dropout=None, stage1_weights=False, dataset=None, log_dir=None, test=False, *args):
    
    print('Loading Scratch ResNet 50 Feature Model.')
    resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], use_fc=use_fc, dropout=None)

    resnet50 = load_model(resnet50, pretrain='/data1/pretrained/resnet50-0676ba61.pth')

    return resnet50
