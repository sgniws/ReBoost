"""Boosted Late Fusion: Sustained Boosting + Adaptive Classifier Assignment for multimodal segmentation."""
import torch
from torch import nn

from models.nnunet import UNet


class ConfigurableSegHead(nn.Module):
    """Configurable segmentation head — corresponds to paper's Configurable Classifier."""

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.private = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor, shared_head: nn.Module) -> torch.Tensor:
        """
        Args:
            features: (B, in_channels, D, H, W) — decoder features
            shared_head: Conv3d(hidden_channels -> 3, k=1) — shared layer
        Returns:
            (B, 3, D, H, W) — segmentation logits
        """
        h = self.private(features)
        return shared_head(h)


class BoostedLateFusion(nn.Module):
    """
    Sustained Boosting + Adaptive Classifier Assignment for segmentation.
    Each modality has one shared UNet backbone and multiple configurable seg heads.
    """

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
        head_hidden_channels=8,
        max_heads_per_modality=10,
    ):
        super().__init__()
        if n_features_per_stage is None:
            n_features_per_stage = [8, 16, 32, 64, 80, 80]
        if kernel_size is None:
            kernel_size = [[3, 3, 3]] * 6
        if strides is None:
            strides = [[1, 1, 1], *[[2, 2, 2]] * 5]

        self.n_experts = n_experts
        self.head_hidden_channels = head_hidden_channels
        self.max_heads_per_modality = max_heads_per_modality

        # 4 independent UNet backbones (encoder + decoder; seg_layers output unused in forward)
        self.backbones = nn.ModuleList([
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

        # Shared head: all modalities, all heads share this layer
        self.shared_head = nn.Conv3d(head_hidden_channels, 3, kernel_size=1, bias=True)

        # Decoder last stage output channels = n_features_per_stage[0] = 8
        feat_channels = n_features_per_stage[0]
        # One head per modality at init
        self.modality_heads = nn.ModuleList([
            nn.ModuleList([
                ConfigurableSegHead(in_channels=feat_channels, hidden_channels=head_hidden_channels)
            ])
            for _ in range(n_experts)
        ])

    def add_head(self, modality_idx: int) -> bool:
        """Add a new segmentation head for the given modality. Returns True if added."""
        if len(self.modality_heads[modality_idx]) >= self.max_heads_per_modality:
            return False
        feat_channels = self.backbones[0].n_features_per_stage[0]
        new_head = ConfigurableSegHead(
            in_channels=feat_channels,
            hidden_channels=self.head_hidden_channels,
        )
        device = next(self.parameters()).device
        new_head = new_head.to(device)
        self.modality_heads[modality_idx].append(new_head)
        return True

    def get_num_heads(self):
        """Return list of head counts per modality."""
        return [len(heads) for heads in self.modality_heads]

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 4, D, H, W) — 4 MRI modalities
        Returns:
            dict:
                'output': (B, 3, D, H, W) — fused prediction for inference
                'modality_all_logits': list of 4 Tensors (B, 3, D, H, W)
                'modality_head_logits': list of 4 lists of Tensors (B, 3, D, H, W)
        """
        modality_all_logits = []
        modality_head_logits = []

        for m_idx in range(self.n_experts):
            x_m = x[:, m_idx : m_idx + 1, ...]  # (B, 1, D, H, W)
            _, decoder_feat = self.backbones[m_idx](x_m, return_features=True)  # (B, 8, D, H, W)

            head_logits_list = []
            for head in self.modality_heads[m_idx]:
                logits = head(decoder_feat, self.shared_head)  # (B, 3, D, H, W)
                head_logits_list.append(logits)

            combined = torch.stack(head_logits_list, dim=0).sum(dim=0)  # (B, 3, D, H, W)
            modality_all_logits.append(combined)
            modality_head_logits.append(head_logits_list)

        output = torch.stack(modality_all_logits, dim=0).mean(dim=0)  # (B, 3, D, H, W)
        return {
            "output": output,
            "modality_all_logits": modality_all_logits,
            "modality_head_logits": modality_head_logits,
        }
