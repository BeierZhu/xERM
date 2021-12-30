import os
import argparse
import pprint
from data import dataloader
from run_networks import model
import warnings
import yaml
from utils import source_import, update

data_root = {'ImageNet': '/data1/ILSVRC/Data/CLS-LOC',
             'Places': '/data1/Places365',
             'CIFAR10': '/data2/CIFAR10',
             'CIFAR100': '/data2/CIFAR100',
             'iNaturalist18': '/data1/iNaturalist2018'}

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None, type=str)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--save_feature', default=False, action='store_true')
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--xERM', default=False, action='store_true')
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# ============================================================================
# Random Seed
import torch
import random
if args.seed:
    print('=======> Using Fixed Random Seed <========')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

# ============================================================================
# LOAD CONFIGURATIONS
with open(args.cfg) as f:
    config = yaml.load(f)
config = update(config, args)

test_mode = args.test
save_mode = args.save_feature  # only in eval
training_opt = config['training_opt']
dataset = training_opt['dataset']

if not os.path.isdir(training_opt['log_dir']):
    os.makedirs(training_opt['log_dir'])

print('Loading dataset from: %s' % data_root[dataset.rstrip('_LT')])
pprint.pprint(config)


# ============================================================================
# TRAINING
if not args.test and not args.xERM:
    # during training, different sampler may be applied
    sampler_defs = training_opt['sampler']
    if sampler_defs:
        if sampler_defs['type'] == 'ClassAwareSampler':
            sampler_dic = {
                'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                'params': {'num_samples_cls': sampler_defs['num_samples_cls']}
            }
        elif sampler_defs['type'] in ['MixedPrioritizedSampler',
                                      'ClassPrioritySampler']:
            sampler_dic = {
                'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                'params': {k: v for k, v in sampler_defs.items() \
                           if k not in ['type', 'def_file']}
            }
    else:
        sampler_dic = None

    # generated sub-datasets all have test split
    splits = ['train', 'val']
    if dataset not in ['iNaturalist18', 'ImageNet']:
        splits.append('test')
    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                    dataset=dataset, phase=x, 
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=sampler_dic,
                                    num_workers=training_opt['num_workers'],
                                    top_k_class=training_opt['top_k'] if 'top_k' in training_opt else None,
                                    cifar_imb_ratio=training_opt['cifar_imb_ratio'] if 'cifar_imb_ratio' in training_opt else None,)
            for x in splits}

    training_model = model(config, data, test=False)
    training_model.train()

# ============================================================================
# TESTING
elif args.test and not args.xERM:
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data",
                            UserWarning)
    print('Under testing phase, we load training data simply to calculate training data number for each class.')

    if 'iNaturalist' in training_opt['dataset']:
        splits = ['train', 'val']
        test_split = 'val'
    else:
        splits = ['train', 'val', 'test']
        test_split = 'test'
    if 'ImageNet' == training_opt['dataset']:
        splits = ['train', 'val']
        test_split = 'val'

    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                    dataset=dataset, phase=x,
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=None, 
                                    num_workers=training_opt['num_workers'],
                                    top_k_class=training_opt['top_k'] if 'top_k' in training_opt else None,
                                    shuffle=True,
                                    cifar_imb_ratio=training_opt['cifar_imb_ratio'] if 'cifar_imb_ratio' in training_opt else None,)
            for x in splits}
    
    training_model = model(config, data, test=True)
    # load checkpoints
    training_model.load_model(training_opt['log_dir'])
    training_model.eval(phase=test_split, save_feat=save_mode)
    # training_model.save_mean_embedding()
elif not args.test and args.xERM:
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data",
                            UserWarning)
    print('Under testing phase, we load training data simply to calculate training data number for each class.')

    if 'iNaturalist' in training_opt['dataset']:
        splits = ['train', 'val']
        test_split = 'val'
    else:
        splits = ['train', 'val', 'test']
        test_split = 'test'
    if 'ImageNet' == training_opt['dataset']:
        splits = ['train', 'val']
        test_split = 'val'

    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                    dataset=dataset, phase=x,
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=None, 
                                    num_workers=training_opt['num_workers'],
                                    top_k_class=training_opt['top_k'] if 'top_k' in training_opt else None,
                                    shuffle=True,
                                    cifar_imb_ratio=training_opt['cifar_imb_ratio'] if 'cifar_imb_ratio' in training_opt else None,)
            for x in splits}
    
    training_model = model(config, data, test=True)
    # load checkpoints
    training_model.load_model(training_opt['log_dir'])
    print('=> Start xERM process .......')
    training_model.xERM_train()
elif args.test and args.xERM:
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data",
                            UserWarning)
    print('Under testing phase, we load training data simply to calculate training data number for each class.')

    if 'iNaturalist' in training_opt['dataset']:
        splits = ['train', 'val']
        test_split = 'val'
    else:
        splits = ['train', 'val', 'test']
        test_split = 'test'
    if 'ImageNet' == training_opt['dataset']:
        splits = ['train', 'val']
        test_split = 'val'

    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                    dataset=dataset, phase=x,
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=None, 
                                    num_workers=training_opt['num_workers'],
                                    top_k_class=training_opt['top_k'] if 'top_k' in training_opt else None,
                                    shuffle=True,
                                    cifar_imb_ratio=training_opt['cifar_imb_ratio'] if 'cifar_imb_ratio' in training_opt else None,)
            for x in splits}
    
    training_model = model(config, data, test=True)
    # load checkpoints
    training_model.load_model(training_opt['log_dir'])
    print('=> Start xERM evaluation .......')
    training_model.xERM_eval()
        
print('='*25, ' ALL COMPLETED ', '='*25)
