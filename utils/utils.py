# -*- coding: utf-8 -*-
"""
Created on 3/09/2020 8:47 pm

@author: Soan Duong, UOW
"""
# Standard library imports
import os
import numpy as np
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# Third party imports
import torch
from torch.utils.data import DataLoader
from monai.data import CacheDataset

# Local application imports
from utils.datasets import ImageDataset, NiiDataset


def init_seeds(seed):
    """

    :param seed:
    :return:
    """
    # Setting seeds
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
def set_devices(device_ids):
    """
    restrict visible device
    :param device_ids: device ids start at 0
    """
    assert len(device_ids) > 0, "there must be at least 1 id"

    os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
    
    
def prepare_device(n_gpu_use=1):
    """
    Setup GPU device if it is available, move the model into the configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There\'s no GPU available on this machine,"
            "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            "Warning: The number of GPU\'s configured to use is {}, but only {} are available "
            "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def get_experiment_dataloaders(cfg):
    """
    Get and return the train, validation, and test dataloaders for the experiment.
    :param cfg: dict that contains the required settings for the dataloaders
                (dataset_dir and train_txtfiles)
    :return: train, validation, and test dataloaders
    """
    m_params = cfg['model_params']
    t_params = cfg['train_params']

    # Generate the train dataloader
    csv_list = [t_params['main_csv']]
    for k in range(t_params['n_incremental']):
        csv_list.append(f'{t_params["append_csv_dir"]}/{k+1}.csv')
    dataset = ImageDataset(csv_list, phase='train', 
                           root_dir=t_params['dataset_dir'],
                           input_shape=m_params['input_shape'])
    
    # Seperate the dataset into train set and val set
    n_train = np.int(0.9 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
    print('No. training samples: ', len(train_set), n_train)
    print('No. validation samples: ', len(val_set), len(dataset) - n_train)
    
    # Generate the dataloader
    train_dataloader = DataLoader(train_set, batch_size=t_params['train_batch_size'],
                                  pin_memory=True, shuffle=True,
                                  num_workers=t_params['num_workers'])

    val_dataloader = DataLoader(val_set, batch_size=t_params['train_batch_size'],
                                pin_memory=True, shuffle=True,
                                num_workers=t_params['num_workers'])

    # Return the dataloaders
    return train_dataloader, val_dataloader


def get_nii_dataloaders(cfg):
    """
    Get and return the train, validation, and test dataloaders for the experiment.
    :param cfg: dict that contains the required settings for the dataloaders
                (dataset_dir and train_txtfiles)
    :return: train, validation, and test dataloaders
    """
    m_params = cfg['model_params']
    t_params = cfg['train_params']
    t_params['n_slices'] = m_params['in_channels']
    # Generate the train dataloader
    dataset = NiiDataset(**t_params)
    
    # Seperate the dataset into train set and val set
    n_train = np.int(0.9 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
    
    if cfg['debug_mode'] == 1:
#         train_ds, val_ds = torch.utils.data.random_split(dataset, [100, 100])
        train_ds = val_ds
        
    print('No. training samples: ', len(train_ds), n_train)
    print('No. validation samples: ', len(val_ds), len(dataset) - n_train)
    
    train_ds = CacheDataset(train_ds, num_workers=t_params['num_workers_to_cache'], 
                            cache_rate=1.0, transform=None)
    train_dataloader = DataLoader(train_ds, batch_size=t_params['train_batch_size'], 
                                  num_workers=t_params['num_workers_from_cache'], 
                                  pin_memory=torch.cuda.is_available(), shuffle=True)
    
    val_ds = CacheDataset(val_ds, num_workers=t_params['num_workers_to_cache'], 
                          cache_rate=1.0, transform=None)
    val_dataloader = DataLoader(val_ds, batch_size=t_params['train_batch_size'], 
                                num_workers=t_params['num_workers_from_cache'], 
                                pin_memory=torch.cuda.is_available(), shuffle=True)
#     # Generate the dataloader
#     train_dataloader = DataLoader(train_set, batch_size=t_params['train_batch_size'],
#                                   pin_memory=True, shuffle=True,
#                                   num_workers=t_params['num_workers'])

#     val_dataloader = DataLoader(val_set, batch_size=t_params['train_batch_size'],
#                                 pin_memory=True, shuffle=True,
#                                 num_workers=t_params['num_workers'])

    # Return the dataloaders
    return train_dataloader, val_dataloader


def init_obj(module_name, module_args, module, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in config, and returns the
    instance initialized with corresponding arguments given.
    `object = config.init_obj('name', module, a, b=1)`
    is equivalent to
    `object = module.name(a, b=1)`
    """
    assert all([k not in module_args for k in
                kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)


def create_exp_dir(path, visual_folder=False):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        if visual_folder is True:
            os.mkdir(path + '/visual')  # for visual results
    else:
        print("DIR already existed.")
    print('Experiment dir : {}'.format(path))


def get_metrics(y_pr, y_gt, labels):
    """
    Compute performance metrics of y_pr and y_gt
    Args:
        y_pr: 2D array of size (batchsize, n_classes)
        y_gt: 1D array of size (batchsize,)
        labels: list of labels of the classification problem
    Returns: dictionary of metrics:
    """

    
    if len(labels) == 2:
        # Get the prob. of label-1 class
        y_pr = y_pr[:, 1]
        auc = roc_auc_score(y_true=y_gt, y_score=y_pr)

        # Get the output labels of the y_pr
        threshold = 0.5
        y_pr[y_pr >= threshold] = 1.0
        y_pr[y_pr < threshold] = 0.0
        accuracy = accuracy_score(y_true=y_gt, y_pred=y_pr)
        precision = precision_score(y_true=y_gt, y_pred=y_pr, pos_label=1, average='binary')
        recall = recall_score(y_true=y_gt, y_pred=y_pr, pos_label=1, average='binary')
        f1_score = fbeta_score(y_true=y_gt, y_pred=y_pr, beta=1, pos_label=1, average='binary')
        f2_score = fbeta_score(y_true=y_gt, y_pred=y_pr, beta=2, pos_label=1, average='binary')

    else:
        # Compute the one-hot coding of the y-gt
        try: 
            y_onehot = np.zeros(y_pr.shape)
            for k in range(len(y_gt)):
                y_onehot[k, y_gt[k]] = 1
            auc = roc_auc_score(y_true=y_onehot, y_score=y_pr)
        
        except Exception: # error when not all classes presented in y_gt
            auc = 0

        # Get the output labels of the y_pr
        y_pr = np.argmax(y_pr, axis=1)
        accuracy = accuracy_score(y_true=y_gt, y_pred=y_pr)
        precision = precision_score(y_true=y_gt, y_pred=y_pr, labels=labels, average='macro')
        recall = recall_score(y_true=y_gt, y_pred=y_pr, pos_label=1, labels=labels, average='macro')
        f1_score = fbeta_score(y_true=y_gt, y_pred=y_pr, beta=1, labels=labels, average='macro')
        f2_score = fbeta_score(y_true=y_gt, y_pred=y_pr, beta=1, labels=labels, average='macro')

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1_score': f1_score, 'f2_score': f2_score, 'auc': auc}
