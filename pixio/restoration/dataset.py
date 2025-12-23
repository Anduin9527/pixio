
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import numpy as np
import random
import torch

from .dataset_utils import (
    FileClient, imfrombytes, img2tensor, padding, 
    paired_paths_from_folder, paired_paths_from_lmdb, 
    paired_random_crop, augment
)

class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type (disk/lmdb) and other kwarg.
            filename_tmpl (str): Template for each filename. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation.
            scale (int): Scale factor.
            phase (str): 'train' or 'val'.
            mean (list): Normalization mean.
            std (list): Normalization std.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt.get('dataset_enlarge_ratio', 1) > 1:
            self.paths *= self.opt['dataset_enlarge_ratio']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt.get('scale', 1)

        # Load gt and lq images
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # Augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # Padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # Random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            
            # Flip, rotation
            img_gt, img_lq = augment(
                [img_gt, img_lq], 
                self.opt.get('use_flip', True),
                self.opt.get('use_rot', True)
            )

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        
        # Normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
