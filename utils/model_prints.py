import torchinfo

from networks.ddrnet import *
from create_network import *

device = 'cuda'
architecture = 'ddrnet'
dataset = 'nyuv2'
if dataset == 'nyuv2':
    tasks = {'seg': 13, 'depth': 1, 'normal': 3}
else:
    sim_warehouse_tasks = {'seg': 23, 'depth': 1, 'normal': 3}

if architecture == 'split':
    model = MTLDeepLabv3(tasks).to(device)
elif architecture == 'mtan':
    model = MTANDeepLabv3(tasks).to(device)
elif architecture == 'ddrnet':
    model = DualResNetMTL(BasicBlock, [2, 2, 2, 2], tasks, dataset, planes=32, spp_planes=128, head_planes=64).to(device)
else:
    raise ValueError('Architecture not supported')
    
print(torchinfo.summary(model, (3, 288, 384), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose = 0))