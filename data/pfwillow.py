r"""PF-WILLOW dataset"""
import os

import pandas as pd
import numpy as np
import torch

from .dataset import CorrespondenceDataset


class PFWillowDataset(CorrespondenceDataset):
    r"""Inherits CorrespondenceDataset"""
    def __init__(self, benchmark, datapath, thres, device, split, augmentation, feature_size):
        r"""PF-WILLOW dataset constructor"""
        super(PFWillowDataset, self).__init__(benchmark, datapath, thres, device, split, augmentation, feature_size)

        self.train_data = pd.read_csv(self.spt_path)
        self.src_imnames = np.array(self.train_data.iloc[:, 0])
        self.trg_imnames = np.array(self.train_data.iloc[:, 1])
        self.src_kps = self.train_data.iloc[:, 2:22].values
        self.trg_kps = self.train_data.iloc[:, 22:].values
        self.cls = ['car(G)', 'car(M)', 'car(S)', 'duck(S)',
                    'motorbike(G)', 'motorbike(M)', 'motorbike(S)',
                    'winebottle(M)', 'winebottle(wC)', 'winebottle(woC)']
        self.cls_ids = list(map(lambda names: self.cls.index(names.split('/')[1]), self.src_imnames))
        self.src_imnames = list(map(lambda x: os.path.join(*x.split('/')[1:]), self.src_imnames))
        self.trg_imnames = list(map(lambda x: os.path.join(*x.split('/')[1:]), self.trg_imnames))

    def __getitem__(self, idx):
        r"""Constructs and return a batch for PF-WILLOW dataset"""
        batch = super(PFWillowDataset, self).__getitem__(idx)
        batch['pckthres'] = self.get_pckthres(batch)

        # batch['src_kpidx'] = self.match_idx(batch['src_kps'], batch['n_pts'])
        # batch['trg_kpidx'] = self.match_idx(batch['trg_kps'], batch['n_pts'])

        batch['flow'] = self.kps_to_flow(batch)
        
        return batch

    def get_pckthres(self, batch):
        r"""Computes PCK threshold"""
        if self.thres == 'bbox':
            return max(batch['src_kps'].max(1)[0] - batch['src_kps'].min(1)[0]).clone()
        if self.thres == 'bbox-kp':
            return max(batch['src_kps'][:, :batch['n_pts']].max(1)[0] - batch['src_kps'][:, :batch['n_pts']].min(1)[0]).clone()
        elif self.thres == 'img':
            return torch.tensor(max(batch['src_img'].size()[1], batch['src_img'].size()[2]))
        else:
            raise Exception('Invalid pck evaluation level: %s' % self.thres)

    def get_points(self, pts_list, idx, org_imsize):
        r"""Returns key-points of an image"""
        point_coords = pts_list[idx, :].reshape(2, 10)
        point_coords = torch.tensor(point_coords.astype(np.float32))
        xy, n_pts = point_coords.size()
        pad_pts = torch.zeros((xy, self.max_pts - n_pts)) - 1
        x_crds = point_coords[0] * (self.imside / org_imsize[0])
        y_crds = point_coords[1] * (self.imside / org_imsize[1])
        kps = torch.cat([torch.stack([x_crds, y_crds]), pad_pts], dim=1)

        return kps, n_pts
