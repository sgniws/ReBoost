from monai.losses import DiceLoss
from torch.nn import Module, BCELoss
from torch import Tensor, sigmoid, numel, maximum, zeros
from torch.nn.functional import binary_cross_entropy


class DiceBCEWithLogitsLoss(Module):
    def __init__(self):
        super(DiceBCEWithLogitsLoss, self).__init__()
        self.dice = DiceLoss(sigmoid=False, reduction='none')
        self.bce = BCELoss(reduction='none')

    def forward(self, preds: Tensor, labels: Tensor, weight=None):
        """Compute Dice Loss & Binary Cross Entropy

        Args:
            preds (Tensor): [B, C, ...]
            labels (Tensor): [B, C, ...]
        """
        probs = sigmoid(preds)
        dice_loss = self.dice(probs, labels)
        bce_loss = self.bce(probs, labels.float())

        if weight is not None:
            dice_loss = dice_loss.mean(dim=[_ for _ in range(1, len(dice_loss.shape))]) * weight
            bce_loss = bce_loss.mean(dim=[_ for _ in range(1, len(bce_loss.shape))]) * weight

        return dice_loss.mean() + bce_loss.mean()
        # return dice_loss, bce_loss.mean()
