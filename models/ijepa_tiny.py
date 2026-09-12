"""Small JEPA-style image encoder for inexpensive petrographic experiments.

This is intentionally a research prototype, not a reproduction of the original
large-scale I-JEPA training configuration.
"""

import copy
import torch
from torch import nn


class PatchEncoder(nn.Module):
    def __init__(self, image_size=224, patch_size=16, dim=192, depth=6, heads=3, mlp_ratio=4.0):
        super().__init__()
        assert image_size % patch_size == 0
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.patch_size = patch_size
        self.dim = dim
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward_tokens(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        return self.norm(self.encoder(x))

    def forward(self, x):
        return self.forward_tokens(x).mean(dim=1)


class IJEPATiny(nn.Module):
    def __init__(self, **encoder_kwargs):
        super().__init__()
        self.context_encoder = PatchEncoder(**encoder_kwargs)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        dim = self.context_encoder.dim
        self.predictor = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    @torch.no_grad()
    def update_target(self, momentum=0.996):
        for online, target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)

    def forward(self, x, target_mask):
        context_tokens = self.context_encoder.forward_tokens(x)
        context_summary = context_tokens.mean(dim=1, keepdim=True)
        prediction = self.predictor(context_summary).expand_as(context_tokens)
        with torch.no_grad():
            target_tokens = self.target_encoder.forward_tokens(x)
        loss = ((prediction[target_mask] - target_tokens[target_mask]) ** 2).mean()
        return loss


def random_target_mask(batch_size, num_tokens, mask_ratio=0.4, device=None):
    n_target = max(1, int(num_tokens * mask_ratio))
    mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
    scores = torch.rand(batch_size, num_tokens, device=device)
    indices = scores.topk(n_target, dim=1).indices
    mask.scatter_(1, indices, True)
    return mask
