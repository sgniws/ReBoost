"""Sustained Boosting loss: fused loss (baseline-aligned) + optional boost terms (ε + ε_all + ε_pre) with normalization."""
from typing import Tuple, Dict, List

import torch
from torch import nn
from torch import Tensor

from loss.dice_bce_loss import DiceBCEWithLogitsLoss


class SustainedBoostingLoss(nn.Module):
    """
    Sustained Boosting loss.
    When all n_heads=1: total_loss = fused_loss (same as baseline).
    When any n_heads>1: total_loss = fused_loss + lambda_boost * (boost_loss_raw / n_boosted).
    """

    def __init__(self, lambda_smooth: float = 0.33, lambda_boost: float = 1.0):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_boost = lambda_boost
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
                'output': (B, 3, D, H, W) — fused logits
                'modality_head_logits': list of 4 lists of Tensors
            labels: (B, 3, D, H, W) — hard labels
        Returns:
            total_loss: scalar Tensor
            loss_details: dict with fused_loss, boost_loss_raw, boost_loss_normalized,
                per-modality epsilon, epsilon_all, epsilon_pre, n_heads
        """
        modality_head_logits = model_output["modality_head_logits"]
        n_modalities = len(modality_head_logits)

        # Fused loss (baseline-aligned): always computed
        fused_loss = self.dice_bce(model_output["output"], labels)

        boost_loss_raw = torch.tensor(0.0, device=labels.device, dtype=labels.dtype)
        loss_details = {"fused_loss": fused_loss.item()}

        for m_idx in range(n_modalities):
            head_logits_list = modality_head_logits[m_idx]
            n_heads = len(head_logits_list)

            # ε_all: combined prediction loss (all heads) — computed for logging for all modalities
            combined_logits = torch.stack(head_logits_list, dim=0).sum(dim=0)
            epsilon_all = self.dice_bce(combined_logits, labels)

            loss_details[f"modality_{m_idx}_epsilon_all"] = epsilon_all.item()
            loss_details[f"modality_{m_idx}_n_heads"] = n_heads

            if n_heads > 1:
                # ε: residual loss (newest head vs residual labels)
                residual_labels = self.compute_residual_labels(labels, head_logits_list)
                newest_probs = torch.sigmoid(head_logits_list[-1])
                epsilon = self.bce(newest_probs, residual_labels)

                # ε_pre: previous heads loss
                pre_logits = torch.stack(head_logits_list[:-1], dim=0).sum(dim=0)
                epsilon_pre = self.dice_bce(pre_logits, labels)

                modality_boost = epsilon + epsilon_all + epsilon_pre
                boost_loss_raw = boost_loss_raw + modality_boost

                loss_details[f"modality_{m_idx}_epsilon"] = (
                    epsilon.item() if isinstance(epsilon, Tensor) else epsilon
                )
                loss_details[f"modality_{m_idx}_epsilon_pre"] = (
                    epsilon_pre.item()
                    if isinstance(epsilon_pre, Tensor)
                    else float(epsilon_pre)
                )
            else:
                loss_details[f"modality_{m_idx}_epsilon"] = 0.0
                loss_details[f"modality_{m_idx}_epsilon_pre"] = 0.0

        n_boosted = sum(
            1 for m_idx in range(n_modalities) if len(modality_head_logits[m_idx]) > 1
        )
        boost_loss_normalized = boost_loss_raw / max(1, n_boosted)
        total_loss = fused_loss + self.lambda_boost * boost_loss_normalized

        loss_details["boost_loss_raw"] = (
            boost_loss_raw.item()
            if isinstance(boost_loss_raw, Tensor)
            else float(boost_loss_raw)
        )
        loss_details["boost_loss_normalized"] = (
            boost_loss_normalized.item()
            if isinstance(boost_loss_normalized, Tensor)
            else float(boost_loss_normalized)
        )
        loss_details["total_loss"] = total_loss.item()
        return total_loss, loss_details
