"""Sustained Boosting loss: ε (residual) + ε_all (combined) + ε_pre (previous heads)."""
from typing import Tuple, Dict, List

import torch
from torch import nn
from torch import Tensor

from loss.dice_bce_loss import DiceBCEWithLogitsLoss


class SustainedBoostingLoss(nn.Module):
    """
    Sustained Boosting loss.
    For each modality: ε (residual) + ε_all (combined) + ε_pre (previous heads).
    """

    def __init__(self, lambda_smooth: float = 0.33):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.dice_bce = DiceBCEWithLogitsLoss()
        self.bce = nn.BCELoss(reduction="mean")

    def compute_residual_labels(
        self, labels: Tensor, head_logits_list: List[Tensor]
    ) -> Tensor:
        """
        Compute residual labels for the newest head.

        Args:
            labels: (B, 3, D, H, W) — hard labels
            head_logits_list: list of Tensors — all head logits for this modality
        Returns:
            (B, 3, D, H, W) — soft residual labels in [0, 1]
        """
        n_heads = len(head_logits_list)
        if n_heads <= 1:
            return labels.float()

        prev_sum = torch.zeros_like(labels, dtype=torch.float32, device=labels.device)
        for j in range(n_heads - 1):
            prev_sum = prev_sum + labels.float() * torch.sigmoid(
                head_logits_list[j].detach()
            )
        residual = labels.float() - self.lambda_smooth * prev_sum
        residual = torch.clamp(residual, min=0.0)
        return residual

    def forward(
        self, model_output: Dict[str, torch.Tensor], labels: Tensor
    ) -> Tuple[Tensor, Dict]:
        """
        Args:
            model_output: output of BoostedLateFusion.forward() — must contain
                'modality_head_logits': list of 4 lists of Tensors
            labels: (B, 3, D, H, W) — hard labels
        Returns:
            total_loss: scalar Tensor
            loss_details: dict with per-modality epsilon, epsilon_all, epsilon_pre, n_heads, total_loss
        """
        modality_head_logits = model_output["modality_head_logits"]
        n_modalities = len(modality_head_logits)

        total_loss = torch.tensor(0.0, device=labels.device, dtype=labels.dtype)
        loss_details = {}

        for m_idx in range(n_modalities):
            head_logits_list = modality_head_logits[m_idx]
            n_heads = len(head_logits_list)

            # ε: residual loss (newest head vs residual labels)
            residual_labels = self.compute_residual_labels(labels, head_logits_list)
            newest_probs = torch.sigmoid(head_logits_list[-1])
            epsilon = self.bce(newest_probs, residual_labels)

            # ε_all: combined prediction loss (all heads)
            combined_logits = torch.stack(head_logits_list, dim=0).sum(dim=0)
            epsilon_all = self.dice_bce(combined_logits, labels)

            # ε_pre: previous heads loss
            if n_heads > 1:
                pre_logits = torch.stack(head_logits_list[:-1], dim=0).sum(dim=0)
                epsilon_pre = self.dice_bce(pre_logits, labels)
            else:
                epsilon_pre = torch.tensor(0.0, device=labels.device, dtype=labels.dtype)

            modality_loss = epsilon + epsilon_all + epsilon_pre
            total_loss = total_loss + modality_loss

            loss_details[f"modality_{m_idx}_epsilon"] = (
                epsilon.item() if isinstance(epsilon, Tensor) else epsilon
            )
            loss_details[f"modality_{m_idx}_epsilon_all"] = epsilon_all.item()
            loss_details[f"modality_{m_idx}_epsilon_pre"] = (
                epsilon_pre.item() if isinstance(epsilon_pre, Tensor) else float(epsilon_pre)
            )
            loss_details[f"modality_{m_idx}_n_heads"] = n_heads

        loss_details["total_loss"] = total_loss.item()
        return total_loss, loss_details
