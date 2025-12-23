
import torch
import os
import shutil
import sys
# Add pretraining to sys.path so that 'layers' can be imported by models_pixio
sys.path.append(os.path.abspath('pretraining'))

from pixio.restoration.dataset import PairedImageDataset
from pixio.restoration.model_wrapper import PixIORestoration
from pretraining import models_pixio

def verify():
    print("Verifying PixIO Restoration Framework...")
    
    # 1. Create dummy data
    os.makedirs('tmp_data/gt', exist_ok=True)
    os.makedirs('tmp_data/lq', exist_ok=True)
    
    # Create 5 dummy images
    import cv2
    import numpy as np
    
    for i in range(5):
        img_gt = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img_lq = cv2.GaussianBlur(img_gt, (5, 5), 0) # Simulate blur, same size
        cv2.imwrite(f'tmp_data/gt/{i:04d}.png', img_gt)
        cv2.imwrite(f'tmp_data/lq/{i:04d}.png', img_lq)
        
    print("Dummy data created.")
    
    # 2. Test Dataset
    opt = {
        'dataroot_gt': 'tmp_data/gt',
        'dataroot_lq': 'tmp_data/lq',
        'io_backend': {'type': 'disk'},
        'phase': 'train',
        'gt_size': 128,
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]
    }
    
    dataset = PairedImageDataset(opt)
    print(f"Dataset length: {len(dataset)}")
    
    sample = dataset[0]
    lq = sample['lq']
    gt = sample['gt']
    
    print(f"Sample shapes - LQ: {lq.shape}, GT: {gt.shape}")
    assert lq.shape == (3, 128, 128)
    assert gt.shape == (3, 128, 128)
    
    # 3. Test Model Wrapper
    print("Initializing model...")
    # Use a smaller model for quick verification if possible, or just standard
    # Using 'pixio_vith16...' but constructing manually to avoid looking up registry if it fails
    
    # We'll use the constructor directly to be safe and strictly control params
    from pretraining.models_pixio import PixioViT
    base_model = PixioViT(
        img_size=128, # Match crop size
        embed_dim=192, depth=2, num_heads=3, # Tiny model for speed
        decoder_embed_dim=192, decoder_depth=2, decoder_num_heads=3
    )
    
    model = PixIORestoration(base_model)
    
    # Forward pass
    print("Running forward pass...")
    input_tensor = lq.unsqueeze(0) # Batch dim
    output = model(input_tensor)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 3, 128, 128)
    
    print("Verification Successful!")
    
    # Cleanup
    shutil.rmtree('tmp_data')

if __name__ == '__main__':
    verify()
