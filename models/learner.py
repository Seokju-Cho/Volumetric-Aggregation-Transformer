
import torch.nn as nn
import torch.nn.functional as F

from .ours import OurModel


class VATLearner(nn.Module):
    def __init__(self, inch):
        super(VATLearner, self).__init__()

        self.ours = OurModel(inch)

    def forward(self, hypercorr_pyramid, query_feats, support_mask):
        return self.ours(hypercorr_pyramid, query_feats, support_mask)