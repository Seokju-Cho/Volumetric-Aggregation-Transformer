
import torch.nn as nn
import torch.nn.functional as F

from model.ours import OurModel


class VATLearner(nn.Module):
    def __init__(self, cfg, inch):
        super(VATLearner, self).__init__()

        if cfg.TRAIN.BENCHMARK == 'pascal':
            feature_affinity = (False, True, True)
        elif cfg.TRAIN.BENCHMARK == 'coco':
            feature_affinity = (False, False, True)
        elif cfg.TRAIN.BENCHMARK == 'fss':
            feature_affinity = (True, True, True)

        self.ours = OurModel(inch, feature_affinity)

    def forward(self, hypercorr_pyramid, query_feats, support_mask):
        return self.ours(hypercorr_pyramid, query_feats, support_mask)
