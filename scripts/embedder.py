"""DINOv2-small embedding wrapper, CPU-friendly, for tile similarity search.

Also exposes per-patch tokens (not just the whole-image CLS summary) — these come from the
same forward pass at zero extra cost, and are what auto_labeler.py uses to localize *where*
within an accepted tile the matching content actually is.
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-small"


class Embedder:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.eval()

    @torch.no_grad()
    def _forward(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state[0]  # [1 + H*W, D]: CLS token + patch tokens
        cls, patches = hidden[0], hidden[1:]
        cls = cls / cls.norm()
        patches = patches / patches.norm(dim=-1, keepdim=True)
        side = int(round(patches.shape[0] ** 0.5))
        patch_grid = patches.reshape(side, side, -1)
        return cls.numpy().astype("float32"), patch_grid.numpy().astype("float32")

    def embed(self, image_path) -> np.ndarray:
        """Whole-image summary vector (CLS token), L2-normalized — for cosine similarity search."""
        cls, _ = self._forward(image_path)
        return cls

    def embed_with_patches(self, image_path):
        """(cls_vec [D], patch_grid [H,W,D]) — patch_grid retains rough spatial position,
        unlike cls_vec which deliberately summarizes the whole image into one point."""
        return self._forward(image_path)
