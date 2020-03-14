import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0): # Default 2.0
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        loss_contrastive = torch.mean((1-label) * torch.pow(distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))

        return loss_contrastive
