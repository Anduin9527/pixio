import argparse
import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

sys.path.append(os.path.abspath("pretraining"))

import swanlab
import yaml

from pretraining import models_pixio
from .dataset import PairedImageDataset
from .model_wrapper import PixIORestoration
from .utils.options import parse_options, dict2str
from .utils.metrics import calculate_psnr, calculate_ssim
from .dataset_utils import tensor2img
import cv2
import numpy as np
import pyiqa
from torchvision.transforms.functional import to_tensor
from loguru import logger
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt", type=str, required=True, help="Path to option YAML file."
    )
    args = parser.parse_args()

    # Parse options
    opt = parse_options(args.opt)

    # Set seed
    seed = opt.get("manual_seed", 42)
    # Initialize Logger
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    os.makedirs(opt["path"]["experiments_root"], exist_ok=True)
    logger.add(
        os.path.join(opt["path"]["experiments_root"], "train.log"), rotation="10 MB"
    )

    logger.info(dict2str(opt))

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(f"Set manual seed: {seed}")

    # Initialize SwanLab
    logger_opt = opt["logger"]
    swanlab_opt = logger_opt.get(
        "swanlab", {}
    )  # Assuming user might pass None in YAML as ~

    use_swanlab = False
    # Check if swanlab_opt is not None and not empty (simplistic check, depends on how YAML parses `~`)
    # If in YAML: swanlab: ~, then swanlab_opt is None.
    # If in YAML: swanlab: {}, then it's empty dict.
    if (
        swanlab_opt is not None
        and isinstance(swanlab_opt, dict)
        and swanlab_opt.get("project") is not None
    ):
        use_swanlab = True
        swanlab.init(
            project=swanlab_opt.get("project", "pixio-restoration"),
            experiment_name=swanlab_opt.get("experiment_name", opt["name"]),
            config=opt,
            logdir=opt["path"]["experiments_root"],
        )
        logger.info("SwanLab initialized.")
    else:
        logger.info("SwanLab logging disabled (configuration missing or empty).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt["path"]["experiments_root"], exist_ok=True)

    # Dataset
    datasets_opt = opt["datasets"]["train"]
    # Add phase for dataset (implied train)
    datasets_opt["phase"] = "train"

    dataset = PairedImageDataset(datasets_opt)
    dataloader = DataLoader(
        dataset,
        batch_size=datasets_opt["batch_size_per_gpu"],
        shuffle=datasets_opt["use_shuffle"],
        num_workers=datasets_opt["num_worker_per_gpu"],
        pin_memory=True,
    )

    # Validation Dataset
    val_dataloader = None
    if "val" in opt["datasets"]:
        val_opt = opt["datasets"]["val"]
        val_opt["phase"] = "val"
        val_dataset = PairedImageDataset(val_opt)
        val_dataloader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
        )
        logger.info(f"Validation dataset loaded: {len(val_dataset)} samples")

    # Model
    net_opt = opt["network_g"]
    logger.info(f"Creating model: {net_opt['type']}")
    base_model = models_pixio.__dict__[net_opt["type"]]()
    if net_opt.get("pretrained_path"):
        checkpoint = torch.load(net_opt["pretrained_path"], map_location="cpu")
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        base_model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pretrained weights from {net_opt['pretrained_path']}")

    mask_ratio = net_opt.get("mask_ratio", 0.0)
    strategy = net_opt.get("training_strategy", "full")
    lora_opt = net_opt.get("lora", None)

    model = PixIORestoration(
        base_model, mask_ratio=mask_ratio, strategy=strategy, lora_opt=lora_opt
    ).to(device)
    logger.info(f"Model initialized with mask_ratio: {mask_ratio}")

    # Optimization
    train_opt = opt["train"]
    total_iter = train_opt["total_iter"]

    optim_params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            optim_params.append(v)

    optimizer = optim.AdamW(
        optim_params,
        lr=train_opt["optim_g"]["lr"],
        weight_decay=train_opt["optim_g"]["weight_decay"],
        betas=tuple(train_opt["optim_g"]["betas"]),
    )

    # Scheduler
    scheduler_type = train_opt.get("scheduler", {}).get("type", "StepLR")
    warmup_iter = train_opt.get("warmup_iter", -1)

    # Define base scheduler
    if scheduler_type == "CosineAnnealingLR":
        T_max = train_opt["scheduler"].get("T_max", total_iter)
        eta_min = train_opt["scheduler"].get("eta_min", 0)
        base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
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
                if scheduler_type == "CosineAnnealingLR":
                    progress = (current_step - warmup_iter) / max(
                        1, total_iter - warmup_iter
                    )
                    progress = min(max(progress, 0.0), 1.0)  # clip
                    return 0.5 * (1 + math.cos(math.pi * progress))
                return 1.0

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = base_scheduler

    # Loss
    if train_opt["pixel_opt"]["type"] == "L1Loss":
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    start_iter = 0
    epoch = 0

    logger.info(f"Start training for {total_iter} iterations")

    current_iter = 0
    while current_iter < total_iter:
        model.train()
        for i, batch in enumerate(dataloader):
            if current_iter >= total_iter:
                break

            lq = batch["lq"].to(device)
            gt = batch["gt"].to(device)

            optimizer.zero_grad()
            output = model(lq)
            loss = criterion(output, gt) * train_opt["pixel_opt"].get(
                "loss_weight", 1.0
            )
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if current_iter % logger_opt["print_freq"] == 0:
                logger.info(
                    f"Iter [{current_iter}/{total_iter}] Loss: {loss.item():.4f}"
                )
                if use_swanlab:
                    swanlab.log({"train/loss": loss.item()}, step=current_iter)

            if (current_iter + 1) % logger_opt["save_checkpoint_freq"] == 0:
                save_path = os.path.join(
                    opt["path"]["experiments_root"],
                    f"checkpoint_{current_iter + 1}.pth",
                )
                torch.save(
                    {
                        "iter": current_iter,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    save_path,
                )
                print(f"Saved checkpoint to {save_path}")
                logger.info(f"Saved checkpoint to {save_path}")

            # Validation and Visualization
            if val_dataloader:
                is_val_step = (current_iter + 1) % opt["validation"]["val_freq"] == 0
                visual_freq = opt["validation"].get(
                    "visual_freq", opt["validation"]["val_freq"]
                )
                is_visual_step = (
                    opt["validation"].get("save_img", False)
                    and (current_iter + 1) % visual_freq == 0
                )

                if is_val_step or is_visual_step:
                    model.eval()

                    # Full Validation
                    if is_val_step:
                        logger.info(f"Validating at iteration {current_iter + 1}...")
                        val_psnr = 0.0
                        val_ssim = 0.0
                        with torch.no_grad():
                            for val_data in tqdm(
                                val_dataloader,
                                desc=f"Validating Iter {current_iter + 1}",
                            ):
                                val_lq = val_data["lq"].to(device)
                                val_gt = val_data["gt"].to(device)

                                val_output = model(val_lq)

                                # Calculate metrics
                                sr_img = tensor2img(val_output)
                                gt_img = tensor2img(val_gt)

                                val_psnr += calculate_psnr(
                                    sr_img, gt_img, crop_border=0
                                )
                                val_ssim += calculate_ssim(
                                    sr_img, gt_img, crop_border=0
                                )

                        avg_psnr = val_psnr / len(val_dataloader)
                        avg_ssim = val_ssim / len(val_dataloader)

                        logger.info(
                            f"Validation PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}"
                        )
                        if use_swanlab:
                            swanlab.log(
                                {"val/psnr": avg_psnr, "val/ssim": avg_ssim},
                                step=current_iter,
                            )

                    # Single Image Validation
                    if is_visual_step:
                        blur_path = "assets/blur.png"
                        sharp_path = "assets/sharp.png"
                        if os.path.exists(blur_path) and os.path.exists(sharp_path):
                            # Read images
                            blur_img = cv2.imread(blur_path).astype(np.float32) / 255.0
                            sharp_img = (
                                cv2.imread(sharp_path).astype(np.float32) / 255.0
                            )

                            # BGR to RGB
                            blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
                            sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)

                            # To Tensor [1, C, H, W]
                            blur_tensor = (
                                torch.from_numpy(blur_img.transpose(2, 0, 1))
                                .float()
                                .unsqueeze(0)
                                .to(device)
                            )
                            sharp_tensor = (
                                torch.from_numpy(sharp_img.transpose(2, 0, 1))
                                .float()
                                .unsqueeze(0)
                                .to(device)
                            )

                            # Inference
                            with torch.no_grad():
                                sr_tensor = model(blur_tensor)

                            # Metrics using pyiqa
                            try:
                                psnr_func = pyiqa.create_metric("psnr", device=device)
                                ssim_func = pyiqa.create_metric("ssim", device=device)
                                single_psnr = psnr_func(sr_tensor, sharp_tensor).item()
                                single_ssim = ssim_func(sr_tensor, sharp_tensor).item()
                            except Exception as e:
                                print(f"Error calculating pyiqa metrics: {e}")
                                single_psnr = 0.0
                                single_ssim = 0.0

                            logger.info(
                                f"Single Image - PSNR: {single_psnr:.4f}, SSIM: {single_ssim:.4f}"
                            )

                            if use_swanlab:
                                # Convert tensors to numpy for logging (RGB)
                                blur_np = tensor2img(blur_tensor, rgb2bgr=False)
                                sharp_np = tensor2img(sharp_tensor, rgb2bgr=False)
                                sr_np = tensor2img(sr_tensor, rgb2bgr=False)

                                swanlab.log(
                                    {
                                        "single_val/psnr": single_psnr,
                                        "single_val/ssim": single_ssim,
                                        "single_val/visual": [
                                            swanlab.Image(sr_np, caption="Restored")
                                        ],
                                    },
                                    step=current_iter,
                                )
                        else:
                            logger.warning(
                                "Warning: assets/blur.png or assets/sharp.png not found, skipping single image validation."
                            )

                    model.train()  # Switch back to train mode

            current_iter += 1
        epoch += 1


if __name__ == "__main__":
    main()
