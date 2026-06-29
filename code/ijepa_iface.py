"""I-JEPA image encoder (the image-native JEPA). mushroom photo -> latent.

REAL mode uses HF transformers (IJepaModel via AutoModel + AutoImageProcessor); the latent is
the mean-pooled patch embeddings (I-JEPA has no CLS token). Like V-JEPA, I-JEPA latents are
non-decodable -- the energy/abstention logic operates entirely in this representation space.
For the synthetic smoke we never call this (images.py plants latents directly).
"""
from __future__ import annotations

import numpy as np

from config import Cfg, CFG


class IJepaEncoder:
    def __init__(self, cfg: Cfg = CFG, synthetic: bool = True):
        self.cfg = cfg
        self.synthetic = synthetic
        self.model = None
        self.proc = None
        if not synthetic:
            import torch
            from transformers import AutoModel, AutoImageProcessor
            self.torch = torch
            self.proc = AutoImageProcessor.from_pretrained(cfg.ijepa_id)
            self.model = AutoModel.from_pretrained(cfg.ijepa_id).eval()
            h = self.model.config.hidden_size
            assert h == cfg.latent_dim, (
                f"I-JEPA hidden_size={h} but cfg.latent_dim={cfg.latent_dim}; set latent_dim={h}."
            )

    def encode(self, pil_images) -> np.ndarray:
        """list of PIL images -> [N, latent_dim] (mean-pooled patch embeddings)."""
        if self.synthetic:
            return np.random.randn(len(pil_images), self.cfg.syn_latent_dim).astype(np.float32)
        with self.torch.no_grad():
            inputs = self.proc(images=pil_images, return_tensors="pt").to(self.model.device)
            out = self.model(**inputs)
            return out.last_hidden_state.mean(dim=1).cpu().numpy()      # [N, hidden]
