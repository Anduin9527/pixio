
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.abspath('pretraining'))

import swanlab

from pretraining import models_pixio
from .dataset import PairedImageDataset
from .model_wrapper import PixIORestoration

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.abspath('pretraining'))

import swanlab
import yaml

from pretraining import models_pixio
from .dataset import PairedImageDataset
from .model_wrapper import PixIORestoration
from .utils.options import parse_options, dict2str
from .utils.metrics import calculate_psnr, calculate_ssim
from .dataset_utils import tensor2img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()
    
    # Parse options
    opt = parse_options(args.opt)
    
    # Set seed
    seed = opt.get('manual_seed', 42)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Set manual seed: {seed}")
        
    print(dict2str(opt))
    
    # Initialize SwanLab
    logger_opt = opt['logger']
    swanlab_opt = logger_opt.get('swanlab', {})
    swanlab.init(
        project=swanlab_opt.get('project', 'pixio-restoration'),
        experiment_name=swanlab_opt.get('experiment_name', opt['name']),
        config=opt,
        logdir=opt['path']['experiments_root']
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(opt['path']['experiments_root'], exist_ok=True)
    
    # Dataset
    datasets_opt = opt['datasets']['train']
    # Add phase for dataset (implied train)
    datasets_opt['phase'] = 'train'
    
    dataset = PairedImageDataset(datasets_opt)
    dataloader = DataLoader(
        dataset, 
        batch_size=datasets_opt['batch_size_per_gpu'], 
        shuffle=datasets_opt['use_shuffle'], 
        num_workers=datasets_opt['num_worker_per_gpu'], 
        pin_memory=True
    )
    
    # Validation Dataset
    val_dataloader = None
    if 'val' in opt['datasets']:
        val_opt = opt['datasets']['val']
        val_opt['phase'] = 'val'
        val_dataset = PairedImageDataset(val_opt)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        print(f"Validation dataset loaded: {len(val_dataset)} samples")
    
    # Model
    net_opt = opt['network_g']
    print(f"Creating model: {net_opt['type']}")
    base_model = models_pixio.__dict__[net_opt['type']]()
    if net_opt.get('pretrained_path'):
        checkpoint = torch.load(net_opt['pretrained_path'], map_location='cpu')
        base_model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Loaded pretrained weights from {net_opt['pretrained_path']}")
        print(f"Loaded pretrained weights from {net_opt['pretrained_path']}")
        
    mask_ratio = net_opt.get('mask_ratio', 0.0)
    strategy = net_opt.get('training_strategy', 'full')
    lora_opt = net_opt.get('lora', None)
    
    model = PixIORestoration(
        base_model, 
        mask_ratio=mask_ratio,
        strategy=strategy,
        lora_opt=lora_opt
    ).to(device)
    print(f"Model initialized with mask_ratio: {mask_ratio}")
    
    model = PixIORestoration(
        base_model, 
        mask_ratio=mask_ratio,
        strategy=strategy,
        lora_opt=lora_opt
    ).to(device)
    print(f"Model initialized with mask_ratio: {mask_ratio}")
    
    # Optimization
    train_opt = opt['train']
    optim_params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            optim_params.append(v)
            
    optimizer = optim.AdamW(
        optim_params, 
        lr=train_opt['optim_g']['lr'], 
        weight_decay=train_opt['optim_g']['weight_decay'],
        betas=tuple(train_opt['optim_g']['betas'])
    )
    
    
    # Scheduler
    scheduler_type = train_opt.get('scheduler', {}).get('type', 'StepLR')
    warmup_iter = train_opt.get('warmup_iter', -1)
    
    # Define base scheduler
    if scheduler_type == 'CosineAnnealingLR':
        T_max = train_opt['scheduler'].get('T_max', total_iter)
        eta_min = train_opt['scheduler'].get('eta_min', 0)
        base_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        # Fallback or default
        base_scheduler = None
        
    # Wrap with Warmup if needed
    if warmup_iter > 0:
        # Custom LambdaLR for Warmup + Base Scheduler combination is tricky
        # A simpler approach used in basicsr/mmcv is to use a SequentialLR or manually handle warmup
        # Here we implement a manual Warmup + Cosine logic
        def lr_lambda(current_step):
            # Warmup phase
            if current_step < warmup_iter:
                return float(current_step) / float(max(1, warmup_iter))
            # Cosine phase
            else:
                # We need to compute the cosine decay factor relative to the post-warmup progress
                # Standard CosineAnnealingLR handles 'epoch' steps, here we work with 'iter'
                # Let's use the formula directly for clarity if base_scheduler is meant to be Cosine
                if scheduler_type == 'CosineAnnealingLR':
                    progress = (current_step - warmup_iter) / max(1, total_iter - warmup_iter)
                    progress = min(max(progress, 0.0), 1.0) # clip
                    return 0.5 * (1 + math.cos(math.pi * progress))
                return 1.0
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = base_scheduler
    
    # Loss
    if train_opt['pixel_opt']['type'] == 'L1Loss':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    
    total_iter = train_opt['total_iter']
    start_iter = 0
    epoch = 0
    
    print(f"Start training for {total_iter} iterations")
    
    current_iter = 0
    while current_iter < total_iter:
        model.train()
        for i, batch in enumerate(dataloader):
            if current_iter >= total_iter:
                break
                
            lq = batch['lq'].to(device)
            gt = batch['gt'].to(device)
            
            optimizer.zero_grad()
            output = model(lq)
            loss = criterion(output, gt) * train_opt['pixel_opt'].get('loss_weight', 1.0)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            if current_iter % logger_opt['print_freq'] == 0:
                print(f"Iter [{current_iter}/{total_iter}] Loss: {loss.item():.4f}")
                swanlab.log({"train/loss": loss.item()}, step=current_iter)
            
            if (current_iter + 1) % logger_opt['save_checkpoint_freq'] == 0:
                save_path = os.path.join(opt['path']['experiments_root'], f'checkpoint_{current_iter+1}.pth')
                torch.save({
                    'iter': current_iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, save_path)
                print(f"Saved checkpoint to {save_path}")
                
            # Validation
            if val_dataloader and (current_iter + 1) % opt['validation']['val_freq'] == 0:
                print(f"Validating at iteration {current_iter+1}...")
                model.eval()
                val_psnr = 0.0
                val_ssim = 0.0
                with torch.no_grad():
                    for val_data in val_dataloader:
                        val_lq = val_data['lq'].to(device)
                        val_gt = val_data['gt'].to(device)
                        
                        val_output = model(val_lq)
                        
                        # Calculate metrics
                        # Convert to numpy and [0, 255]
                        sr_img = tensor2img(val_output) # You need to implement tensor2img in dataset_utils or import from basicsr
                        gt_img = tensor2img(val_gt)
                        
                        val_psnr += calculate_psnr(sr_img, gt_img, crop_border=0)
                        val_ssim += calculate_ssim(sr_img, gt_img, crop_border=0)
                
                avg_psnr = val_psnr / len(val_dataloader)
                avg_ssim = val_ssim / len(val_dataloader)
                
                print(f"Validation PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
                swanlab.log({"val/psnr": avg_psnr, "val/ssim": avg_ssim}, step=current_iter)
                model.train() # Switch back to train mode

            current_iter += 1
        epoch += 1

if __name__ == '__main__':
    main()
