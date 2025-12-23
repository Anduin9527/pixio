import torch
import torch.nn as nn


class PixIORestoration(nn.Module):
    """Wrapper for PixIO to perform image restoration.
    It disables the masking mechanism (mask_ratio=0) during inference/training.
    """

    """
    """

    def __init__(self, encoder_model, mask_ratio=0.0, strategy="full", lora_opt=None):
        super().__init__()
        self.encoder = encoder_model
        self.mask_ratio = mask_ratio

        self.max_grad_norm_warned = False  # For logging

        # Apply Training Strategy
        self._apply_strategy(strategy, lora_opt)

    def _apply_strategy(self, strategy, lora_opt):
        print(f"Applying training strategy: {strategy}")

        if strategy == "full":
            # Default: all parameters trainable
            for param in self.parameters():
                param.requires_grad = True

        elif strategy == "freeze_encoder":
            # Freeze Encoder, Train Decoder
            # Note: PixioViT structure (encoder is self.encoder, but wait, self.encoder IS the PixioViT which has .blocks (encoder) and .decoder_blocks)
            # We need to be careful. In models_pixio.py, PixioViT contains BOTH encoder blocks and decoder blocks.
            #   self.blocks = nn.ModuleList([...])  <-- Encoder Blocks
            #   self.decoder_blocks = nn.ModuleList([...]) <-- Decoder Blocks

            # Freeze everything first
            for param in self.encoder.parameters():
                param.requires_grad = False

            # Unfreeze decoder parts
            # Decoder components in PixioViT: decoder_embed, mask_token, decoder_pos_embed, decoder_blocks, decoder_norm, decoder_pred
            decoder_modules = [
                self.encoder.decoder_embed,
                self.encoder.mask_token,
                self.encoder.decoder_pos_embed,
                self.encoder.decoder_blocks,
                self.encoder.decoder_norm,
                self.encoder.decoder_pred,
            ]

            for mod in decoder_modules:
                if isinstance(mod, torch.Tensor):  # Parameters like mask_token
                    mod.requires_grad = True
                else:
                    for param in mod.parameters():
                        param.requires_grad = True

            print("  -> Encoder frozen. Decoder trainable.")

        elif strategy == "lora":
            if lora_opt is None:
                raise ValueError("LoRA config is missing for strategy 'lora'")

            try:
                from peft import LoraConfig, get_peft_model
            except ImportError:
                raise ImportError("Please install peft: pip install peft")

            # LoRA typically targets Linear layers.
            # In PixioViT:
            #   Attention: qkv, proj
            #   MLP: fc1, fc2
            # We want to apply LoRA ONLY to the ENCODER blocks usually, or both?
            # Consistent with 'freeze_encoder' logic: we want to adapt the Encoder (via LoRA) and fully train Decoder.
            # OR adapt both?
            #   Option 1: LoRA on Encoder + Full Decoder (Hybrid) -> Best for restoration
            #   Option 2: LoRA on Everything -> Very parameter efficient

            # Given user context: "LoRA on Encoder + Decoder Full Train" recommended.
            # Let's implement Option 1.

            # Step 1: Freeze Encoder
            for param in self.encoder.blocks.parameters():
                param.requires_grad = False
            for param in self.encoder.patch_embed.parameters():
                param.requires_grad = False
            for param in self.encoder.norm.parameters():
                param.requires_grad = False
            # (Global tokens/pos embed also frozen)
            self.encoder.pos_embed.requires_grad = False
            self.encoder.cls_token.requires_grad = False

            # Step 2: Inject LoRA into Encoder
            target_modules = lora_opt.get(
                "target_modules", ["qkv", "proj", "fc1", "fc2"]
            )
            r = lora_opt.get("r", 16)
            alpha = lora_opt.get("alpha", 32)
            dropout = lora_opt.get("dropout", 0.05)

            # Regex to target ONLY encoder blocks?
            # target_modules name matching is usually substring.
            # If we say "qkv", it matches decoder's qkv too.
            # We can use `target_modules` list or regex. To be safe/simple, let's let peft wrap "qkv".
            # BUT we want Decoder to be FULLY trainable (not LoRA).
            # If we wrap Decoder with LoRA, we are restricting it.
            # Strategy: Apply LoRA to whole model, then Unfreeze Decoder parameters completely?
            # Or better: construct a regexp that only matches encoder blocks.
            # Encoder blocks valid names: "blocks.*.qkv", etc.
            # Decoder blocks: "decoder_blocks..."

            # Construct strict target modules list matching structure
            # To simplify, we can use `peft`'s `modules_to_save`.
            # But specific targeting is better.

            # Let's try: Apply LoRA to SPECIFIC patterns "blocks.*.qkv" etc.
            # But target_modules in LoraConfig usually takes suffix.
            # WORKAROUND: We apply LoRA to the whole model finding suffixes.
            # Then we iterating specifically setting Decoder params to requires_grad=True.

            config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                target_modules=target_modules,
                lora_dropout=dropout,
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, config)

            # Now self.encoder is a PeftModel wrapping the original PixioViT
            # PeftModel sets requires_grad=False for everything except LoRA params.

            # Step 3: UNFREEZE Decoder (make it fully trainable)
            # Access original model via self.encoder.base_model.model (Peft structure)
            base_model_ref = self.encoder.base_model.model

            decoder_modules = [
                base_model_ref.decoder_embed,
                base_model_ref.mask_token,
                base_model_ref.decoder_pos_embed,
                base_model_ref.decoder_blocks,
                base_model_ref.decoder_norm,
                base_model_ref.decoder_pred,
            ]

            for mod in decoder_modules:
                if isinstance(mod, torch.Tensor):
                    mod.requires_grad = True
                else:
                    for param in mod.parameters():
                        param.requires_grad = (
                            True  # Force unfreeze for full finetuning of decoder
                        )

            # Also unfreeze LoRA params (they should already be True, but double check)
            # (Peft handles LoRA params)

            trainable_params, all_params = self.encoder.get_nb_trainable_parameters()
            print(f"  -> LoRA applied to Encoder. Decoder fully unfrozen.")
            print(
                f"     Trainable params: {trainable_params:,d} || All params: {all_params:,d} || Trainable%: {100 * trainable_params / all_params:.2f}%"
            )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images (B, C, H, W).
        Returns:
            torch.Tensor: Reconstructed images (B, C, H, W).
        """
        # mask_ratio=0 means no masking, effectively using the whole image
        # The encoder forward returns a tuple (loss, pred, mask) if we call forward()
        # But we want the reconstruction, so we should look at how the encoder works.
        # PixioViT.forward returns loss.
        # We need to call forward_encoder and forward_decoder manually or modify the forward.

        # Using forward_encoder and forward_decoder
        # Using forward_encoder and forward_decoder
        # Support passing mask_ratio, default to 0.0 (no masking)
        mask_ratio = getattr(self, "mask_ratio", 0.0)
        latent, mask, ids_restore = self.encoder.forward_encoder(
            x, mask_ratio=mask_ratio
        )
        pred = self.encoder.forward_decoder(
            latent,
            ids_restore,
            shape=(
                x.shape[2] // self.encoder.patch_embed.patch_size[0],
                x.shape[3] // self.encoder.patch_embed.patch_size[1],
            ),
        )  # [N, L, p*p*3]

        # Pred is [N, L, p*p*3], we need to unpatchify it to [N, 3, H, W]
        # PixioViT model in `models_pixio.py` has `unpatchify` method?
        # Let's check `models_pixio.py` again. It has `patchify` but maybe not `unpatchify`?
        # MAE usually implementation has `unpatchify`.

        return self.unpatchify(
            pred,
            (
                x.shape[2] // self.encoder.patch_embed.patch_size[0],
                x.shape[3] // self.encoder.patch_embed.patch_size[1],
            ),
        )

    def unpatchify(self, x, grid_shape=None):
        """
        x: (N, L, patch_size**2 *3)
        grid_shape: (h, w) of the patch grid
        imgs: (N, 3, H, W)
        """
        p = self.encoder.patch_embed.patch_size[0]
        if grid_shape is None:
            h = w = int(x.shape[1] ** 0.5)
            assert h * w == x.shape[1]
        else:
            h, w = grid_shape
            assert h * w == x.shape[1], (
                f"Grid shape {h}x{w}={h * w} does not match sequence length {x.shape[1]}"
            )

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs
