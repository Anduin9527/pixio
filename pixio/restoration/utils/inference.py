import torch
import torch.nn.functional as F


def pad_img(img, patch_size=16):
    _, _, h, w = img.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        return F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
    return img


def inference_single(model, img, use_shift_ensemble=True, patch_size=16):
    """
    Inference a single image (or tile) with optional Shift Ensemble.
    """
    if not use_shift_ensemble:
        # Standard padding to patch_size multiple
        img_pad = pad_img(img, patch_size)
        with torch.no_grad():
            output = model(img_pad)
        # Crop back
        _, _, h, w = img.shape
        output = output[:, :, :h, :w]
        return output

    # 4-pass Shift Ensemble
    shifts = [
        (0, 0),
        (0, patch_size // 2),
        (patch_size // 2, 0),
        (patch_size // 2, patch_size // 2),
    ]
    sr_accum = None
    count_accum = None

    _, _, h_old, w_old = img.shape

    for shift_y, shift_x in shifts:
        # 1. Shift by padding top-left
        img_shifted = F.pad(img, (shift_x, 0, shift_y, 0), mode="reflect")

        # 2. Pad to multiple of patch_size (bottom-right)
        img_input = pad_img(img_shifted, patch_size)

        # 3. Inference
        with torch.no_grad():
            sr_output = model(img_input)

        # 4. Crop back to shifted size (remove bottom-right padding)
        _, _, h_s, w_s = img_shifted.shape
        sr_output = sr_output[:, :, :h_s, :w_s]

        # 5. Shift back (crop top-left)
        sr_output = sr_output[:, :, shift_y:, shift_x:]

        # 6. Accumulate
        if sr_accum is None:
            sr_accum = torch.zeros_like(sr_output)
            count_accum = torch.zeros_like(sr_output)

        sr_accum += sr_output
        count_accum += 1.0

    return sr_accum / count_accum


def sliding_window_inference(
    model, img, tile_size=None, tile_overlap=32, use_shift_ensemble=True
):
    """
    Perform inference using sliding window (tiling) strategy.

    Args:
        model: The network.
        img: Input tensor (B, C, H, W).
        tile_size: Size of the tile. If None, use the whole image.
        tile_overlap: Overlap between tiles.
        use_shift_ensemble: Whether to use 4-pass shift ensemble within each inference.
    """
    if tile_size is None:
        return inference_single(model, img, use_shift_ensemble)

    b, c, h, w = img.shape
    assert b == 1, "Only batch size 1 is supported for tiling inference"

    tile_size = min(tile_size, h, w)
    assert tile_size % 16 == 0, "Tile size should be a multiple of 16"

    stride = tile_size - tile_overlap
    h_idx_list = list(range(0, h - tile_size, stride)) + [h - tile_size]
    w_idx_list = list(range(0, w - tile_size, stride)) + [w - tile_size]

    E = torch.zeros(b, c, h, w, dtype=img.dtype, device=img.device)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img[..., h_idx : h_idx + tile_size, w_idx : w_idx + tile_size]

            out_patch = inference_single(model, in_patch, use_shift_ensemble)

            E[..., h_idx : h_idx + tile_size, w_idx : w_idx + tile_size].add_(out_patch)
            W[..., h_idx : h_idx + tile_size, w_idx : w_idx + tile_size].add_(1.0)

    return E.div_(W)
