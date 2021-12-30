"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from models.ResNetFeature import *
from utils import *
from os import path
        
def create_model(use_fc=False, dropout=None, stage1_weights=False, dataset=None, log_dir=None, test=False, *args):
    
    print('Loading Scratch ResNet 152 Feature Model.')
    resnet = ResNet(Bottleneck, [3, 8, 36, 3], use_fc=use_fc, dropout=None)

    resnet = init_weights(model=resnet,
                             weights_path='./logs/resnet152.pth',
                             caffe=True)

    print('=> load resnet 152')
    
    return resnet
