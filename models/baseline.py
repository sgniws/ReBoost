"""Baseline Late Fusion: 4 single-channel UNet experts with static equal-weight averaging."""
import torch
from torch import nn

from models.nnunet import UNet


class BaselineLateFusion(nn.Module):
    """Four independent single-channel 3D UNet experts; fusion by static equal-weight average of logits."""

    def __init__(
        self,
        input_channels=1,
        n_classes=3,
        n_stages=6,
        n_features_per_stage=None,
        kernel_size=None,
        strides=None,
        apply_deep_supervision=False,
        n_experts=4,
    ):
        super().__init__()
        if n_features_per_stage is None:
            n_features_per_stage = [8, 16, 32, 64, 80, 80]
        if kernel_size is None:
            kernel_size = [[3, 3, 3]] * 6
        if strides is None:
            strides = [[1, 1, 1], *[[2, 2, 2]] * 5]

        self.n_experts = n_experts
        self.expert_ls = nn.ModuleList([
            UNet(
                input_channels=input_channels,
                n_classes=n_classes,
                n_stages=n_stages,
                n_features_per_stage=n_features_per_stage,
                kernel_size=kernel_size,
                strides=strides,
                apply_deep_supervision=apply_deep_supervision,
            )
            for _ in range(n_experts)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, 4, D, H, W) — 4 MRI modalities (T1, T1ce, T2, FLAIR)
        Returns:
            output: (B, 3, D, H, W) — fused segmentation logits (WT, TC, ET)
        """
        expert_outputs = []
        for modality_idx in range(self.n_experts):
            x_m = x[:, modality_idx : modality_idx + 1, ...]  # (B, 1, D, H, W)
            o_m = self.expert_ls[modality_idx](x_m)  # (B, 3, D, H, W)
            expert_outputs.append(o_m)

        # static equal-weight average in logit space
        output = torch.stack(expert_outputs, dim=0).mean(dim=0)  # (B, 3, D, H, W)
        return output
