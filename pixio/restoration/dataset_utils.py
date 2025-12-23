
import cv2
import numpy as np
import torch
import os
import random
from os import path as osp
from torch.nn import functional as F

# -------------------------------------------------------------------------
# File Client (Disk & LMDB)
# -------------------------------------------------------------------------

class BaseStorageBackend:
    def get(self, filepath):
        pass

class HardDiskBackend(BaseStorageBackend):
    def get(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

class LmdbBackend(BaseStorageBackend):
    def __init__(self, db_paths, client_keys='default', readonly=True, lock=False, readahead=False, **kwargs):
        try:
            import lmdb
        except ImportError:
            raise ImportError('Please install lmdb to enable LmdbBackend. `pip install lmdb`')

        if isinstance(client_keys, str):
            client_keys = [client_keys]
        if isinstance(db_paths, list):
            self.db_paths = [str(v) for v in db_paths]
        elif isinstance(db_paths, str):
            self.db_paths = [str(db_paths)]
        
        self._client = {}
        for client, path in zip(client_keys, self.db_paths):
            self._client[client] = lmdb.open(
                path, readonly=readonly, lock=lock, readahead=readahead, 
                map_size=1024**3, **kwargs) # 1GB map size default, can be adjusted

    def get(self, filepath, client_key):
        filepath = str(filepath)
        client = self._client[client_key]
        with client.begin(write=False) as txn:
            value_buf = txn.get(filepath.encode('ascii'))
        return value_buf

class FileClient:
    _backends = {
        'disk': HardDiskBackend,
        'lmdb': LmdbBackend,
    }

    def __init__(self, backend='disk', **kwargs):
        if backend not in self._backends:
            raise ValueError(f'Backend {backend} is not supported.')
        self.backend = backend
        self.client = self._backends[backend](**kwargs)

    def get(self, filepath, client_key='default'):
        if self.backend == 'lmdb':
            return self.client.get(filepath, client_key)
        else:
            return self.client.get(filepath)

# -------------------------------------------------------------------------
# Data Path Generation
# -------------------------------------------------------------------------

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix, recursive)

def paired_paths_from_folder(folders, keys, filename_tmpl):
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = list(scandir(input_folder, full_path=True))
    gt_paths = list(scandir(gt_folder, full_path=True))
    
    # Simple matching assumes filenames contain the key or are identical structure
    # Here we strictly follow basicsr logic: match by filename
    paths = []
    
    # Build a dict for faster lookup
    gt_dict = {osp.splitext(osp.basename(p))[0]: p for p in gt_paths}
    
    for input_path in input_paths:
        basename = osp.splitext(osp.basename(input_path))[0]
        # Inverse the template logic if needed, but for simplicity we assume direct match 
        # or simplified template matching. 
        # If filename_tmpl is '{}', basenames should match.
        # Check if basename corresponds to a GT
        if basename in gt_dict:
            gt_path = gt_dict[basename]
            paths.append({
                f'{input_key}_path': input_path,
                f'{gt_key}_path': gt_path
            })
    return paths

def paired_paths_from_lmdb(folders, keys):
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith('.lmdb') and gt_folder.endswith('.lmdb')):
        raise ValueError('Folders should be LMDB.')

    with open(osp.join(input_folder, 'meta_info.txt')) as fin:
        input_lmdb_keys = [line.split('.')[0] for line in fin]
    
    # We assume keys are consistent
    paths = []
    for lmdb_key in sorted(input_lmdb_keys):
        paths.append({
            f'{input_key}_path': lmdb_key,
            f'{gt_key}_path': lmdb_key
        })
    return paths

# -------------------------------------------------------------------------
# Image Utils
# -------------------------------------------------------------------------

def imfrombytes(content, float32=False):
    img_np = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if float32:
        img = img.astype(np.float32) / 255.
    return img

def img2tensor(imgs, bgr2rgb=True, float32=True):
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def padding(img_lq, img_gt, gt_size):
    h, w, _ = img_lq.shape
    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    
    if h_pad == 0 and w_pad == 0:
        return img_lq, img_gt

    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    return img_lq, img_gt

def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
         raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x LQ ({h_lq}, {w_lq}).')

    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]
    
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]

    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs

def augment(imgs, hflip=True, rotation=True):
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:
            cv2.flip(img, 1, img)
        if vflip:
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]
    return imgs

def tensor2img(tensor, rgb2bgr=True, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        
        n_dim = _tensor.dim()
        if n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:
                img_np = np.squeeze(img_np, axis=2)
            elif img_np.shape[2] == 3:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
             raise TypeError(f'Only support 3D tensor. But received with dimension: {n_dim}')
            
        img_np = (img_np * 255.0).round()
        img_np = img_np.astype(np.uint8)
        result.append(img_np)
        
    if len(result) == 1:
        result = result[0]
    return result
