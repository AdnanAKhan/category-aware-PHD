import torch.nn as nn
import torch.nn.functional as F
from utils.utils import load_checkpoint
import os
from model.e2e_net import CustomC3D
from pprint import pprint
import torch

model = CustomC3D()
model_dict = model.state_dict()
pre_trained_weight_path = os.path.join( 
    '/home/adnankhan/PycharmProjects/HighlightDetection/pytorch_implementation/experiments/c3d_pre_trained_model', 
    'c3d.pickle')

pretrained_dict = torch.load(pre_trained_weight_path)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 
model.load_state_dict(pretrained_dict)

# load_checkpoint(os.path.join(
#     '/home/adnankhan/PycharmProjects/HighlightDetection/pytorch_implementation/experiments/c3d_pre_trained_model',
#     'c3d.pickle'), model)
print(model)