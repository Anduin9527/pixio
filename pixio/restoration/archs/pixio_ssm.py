import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import sys
import os

# Try importing mamba, handle potential import errors gracefully or assume environment is ready
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None
    print("Warning: mamba_ssm not found. PixIO-SSM will not work without it.")

sys.path.append(os.path.abspath("pretraining"))
from pretraining.models_pixio import PixioViT, pixio_vitb16_enc768x12h12_dec512x32h16

# ================= from EVSSM_arch.py =================


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        import numbers

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight
            + self.bias
        )


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.patch_size = 8
        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.fft = nn.Parameter(
            torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1))
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.GELU()

        self.x_proj = (
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )
        del self.x_proj

        self.x_conv = nn.Conv1d(
            in_channels=(self.dt_rank + self.d_state * 2),
            out_channels=(self.dt_rank + self.d_state * 2),
            kernel_size=7,
            padding=3,
            groups=(self.dt_rank + self.d_state * 2),
        )

        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 1
        x_hwwh = x.view(B, 1, -1, L)
        xs = x_hwwh
        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight
        )
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight
        )
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        return out_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        x = rearrange(x, "b c h w -> b h w c")
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1 = self.forward_core(x)
        y = y1
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)
        out = rearrange(out, "b h w c -> b c h w")
        return out


class EVS(nn.Module):
    def __init__(
        self, dim, ffn_expansion_factor=2.66, bias=False, att=False, idx=0, patch=128
    ):
        super(EVS, self).__init__()
        self.att = att
        self.idx = idx
        if self.att:
            self.norm1 = LayerNorm(dim)
            self.attn = SS2D(d_model=dim)  # Simplified: removed unused args
        self.norm2 = LayerNorm(dim)
        self.ffn = EDFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        if self.att:
            if self.idx % 2 == 1:
                x = torch.flip(x, dims=(-2, -1)).contiguous()
            if self.idx % 2 == 0:
                x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()
            x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        return self.body(x)


# ================= PixIO-SSM Wrapper =================


class PixIO_SSM(nn.Module):
    def __init__(
        self,
        base_model_name="pixio_vitb16_enc768x12h12_dec512x32h16",
        decoder_dim=384,
        out_channels=3,
        pretrained_path=None,
    ):
        super(PixIO_SSM, self).__init__()

        # 1. Initialize PixIO Encoder
        # We instantiate the full ViT but we will throw away its decoder
        self.base_model = pixio_vitb16_enc768x12h12_dec512x32h16()
        self.embed_dim = self.base_model.embed_dim  # 768
        self.patch_size = self.base_model.patch_embed.patch_size[0]  # 16

        # Load Weights for Encoder
        if pretrained_path:
            self.load_encoder_weights(pretrained_path)

        # Freeze Encoder? Config will handle this in PixIORestoration Wrapper if needed
        # But here we just define architecture.

        # 2. Define SSM Decoder (Progressive Upsampling)
        # 1/16 (768) -> 1/8 (decoder_dim) -> 1/4 -> 1/2 -> 1

        # Project 768 -> decoder_dim
        self.proj_in = nn.Conv2d(self.embed_dim, decoder_dim, 1)

        # Stage 1: 1/16 -> 1/8
        self.up1 = Upsample(
            decoder_dim * 2
        )  # Input 2C -> Output C. So If input is decoder_dim, Upsample expects 2*output.
        # Wait, Upsample(n_feat) takes n_feat and produces n_feat//2.
        # So we want output to be decoder_dim. Input should be 2*decoder_dim.
        # But we project to decoder_dim first.
        # Implementation Detail:
        # E.g. Latent (768).
        # Level 0 (Bottleneck): EVS Blocks at 1/16?

        self.bottleneck = nn.Sequential(
            *[EVS(dim=decoder_dim, idx=i, att=True) for i in range(2)]
        )

        # Upsample 1: 1/16 -> 1/8.
        # Input: decoder_dim. Output should be decoder_dim/2?
        # EVSSM decays dim. Let's maintain dim or decay?
        # Let's say: 384 -> 192 -> 96 -> 48 -> Out.

        dims = [384, 192, 96, 48]  # Example
        # Adjust based on decoder_dim input
        d = decoder_dim
        self.dims = [d, d // 2, d // 4, d // 8]

        # Override proj_in
        self.proj_in = nn.Conv2d(self.embed_dim, self.dims[0], 1)

        # 1/16 -> 1/8
        self.up1 = Upsample(
            self.dims[0]
        )  # 384 -> 192 (Upsample definition: Conv(n, n//2))
        self.stage1 = nn.Sequential(
            *[EVS(dim=self.dims[1], idx=i, att=True) for i in range(2)]
        )

        # 1/8 -> 1/4
        self.up2 = Upsample(self.dims[1])  # 192 -> 96
        self.stage2 = nn.Sequential(
            *[EVS(dim=self.dims[2], idx=i, att=True) for i in range(2)]
        )

        # 1/4 -> 1/2
        self.up3 = Upsample(self.dims[2])  # 96 -> 48
        self.stage3 = nn.Sequential(
            *[EVS(dim=self.dims[3], idx=i, att=True) for i in range(2)]
        )

        # 1/2 -> 1/1
        self.up4 = Upsample(self.dims[3])  # 48 -> 24
        self.stage4 = nn.Sequential(
            *[EVS(dim=self.dims[3] // 2, idx=i, att=True) for i in range(2)]
        )

        self.tail = nn.Conv2d(self.dims[3] // 2, out_channels, 3, 1, 1)

    def load_encoder_weights(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Filter only encoder keys and non-decoder keys
        # Encoder keys usually start with nothing (blocks., patch_embed.)
        # Decoder keys start with 'decoder_'
        encoder_dict = {}
        for k, v in state_dict.items():
            if not k.startswith("decoder") and not k == "mask_token":
                encoder_dict[k] = v

        msg = self.base_model.load_state_dict(encoder_dict, strict=False)
        print(f"Loaded Encoded Weights: {msg}")

    def forward_features(self, x):
        # We need latent features from PixIO Encoder
        # Use existing method forward_encoder, but mask_ratio=0
        # forward_encoder returns: x (latent), mask, ids_restore
        # x shape: [N, L, D] including CLS token

        latent, _, _ = self.base_model.forward_encoder(x, mask_ratio=0.0)

        # Remove CLS tokens?
        n_cls = self.base_model.n_cls_tokens
        latent = latent[:, n_cls:, :]  # [N, H*W, D]

        # Unpatchify to 2D
        B, L, D = latent.shape
        H = W = int(math.sqrt(L))
        # assert H == x.shape[2] // 16

        feat = latent.transpose(1, 2).view(B, D, H, W)
        return feat

    def forward(self, x):
        # Encoder
        # lat: [B, 768, 16, 16] (if input 256)
        lat = self.forward_features(x)

        # Decoder
        y = self.proj_in(lat)  # [B, 384, 16, 16]
        y = self.bottleneck(y)

        y = self.up1(y)  # [B, 192, 32, 32]
        y = self.stage1(y)

        y = self.up2(y)  # [B, 96, 64, 64]
        y = self.stage2(y)

        y = self.up3(y)  # [B, 48, 128, 128]
        y = self.stage3(y)

        y = self.up4(y)  # [B, 24, 256, 256]
        y = self.stage4(y)

        out = self.tail(y)
        return out + x  # Residual learning (Global Skip)
