"""For quantitative evaluation of DHPF"""
from skimage import draw
import numpy as np
import torch

from . import utils


class Evaluator:
    r"""Computes evaluation metrics of PCK, LT-ACC, IoU"""
    @classmethod
    def initialize(cls, benchmark, alpha=0.1):
        if benchmark == 'caltech':
            cls.eval_func = cls.eval_mask_transfer
        else:
            cls.eval_func = cls.eval_kps_transfer
        cls.alpha = alpha

    @classmethod
    def evaluate(cls, prd_kps, batch):
        r"""Compute evaluation metric"""
        return cls.eval_func(prd_kps, batch)

    @classmethod
    def eval_kps_transfer(cls, prd_kps, batch):
        r"""Compute percentage of correct key-points (PCK) based on prediction"""

        easy_match = {'src': [], 'trg': [], 'dist': []}
        hard_match = {'src': [], 'trg': []}

        pck = []
        for idx, (pk, tk) in enumerate(zip(prd_kps, batch['src_kps'])):
            thres = batch['pckthres'][idx]
            npt = batch['n_pts'][idx]
            _, correct_ids, _ = cls.classify_prd(pk[:, :npt], tk[:, :npt], thres)

            pck.append((len(correct_ids) / npt.item()) * 100)

        eval_result = {'pck': pck}

        return eval_result

    @classmethod
    def eval_kps_transfer_with_correct(cls, prd_kps, batch):
        r"""Compute percentage of correct key-points (PCK) based on prediction"""

        easy_match = {'src': [], 'trg': [], 'dist': []}
        hard_match = {'src': [], 'trg': []}

        pck = []
        correct_id_list = []
        for idx, (pk, tk) in enumerate(zip(prd_kps, batch['src_kps'])):
            thres = batch['pckthres'][idx]
            npt = batch['n_pts'][idx]
            _, correct_ids, _ = cls.classify_prd(pk[:, :npt], tk[:, :npt], thres)
            correct_id_list.append(correct_ids)

            pck.append((len(correct_ids) / npt.item()) * 100)

        eval_result = {'pck': pck}

        return eval_result, correct_id_list

    @classmethod
    def eval_mask_transfer(cls, prd_kps, batch):
        r"""Compute LT-ACC and IoU based on transferred points"""

        ltacc = []
        iou = []

        for idx, prd in enumerate(prd_kps):
            trg_n_pts = (batch['trg_kps'][idx] > 0)[0].sum()
            prd_kp = prd[:, :batch['n_pts'][idx]]
            trg_kp = batch['trg_kps'][idx][:, :trg_n_pts]

            imsize = list(batch['trg_img'].size())[2:]
            trg_xstr, trg_ystr = cls.pts2ptstr(trg_kp)
            trg_mask = cls.ptstr2mask(trg_xstr, trg_ystr, imsize[0], imsize[1])
            prd_xstr, pred_ystr = cls.pts2ptstr(prd_kp)
            prd_mask = cls.ptstr2mask(prd_xstr, pred_ystr, imsize[0], imsize[1])

            ltacc.append(cls.label_transfer_accuracy(prd_mask, trg_mask))
            iou.append(cls.intersection_over_union(prd_mask, trg_mask))

        eval_result = {'ltacc': ltacc,
                       'iou': iou}

        return eval_result

    @classmethod
    def classify_prd(cls, prd_kps, trg_kps, pckthres):
        r"""Compute the number of correctly transferred key-points"""
        l2dist = (prd_kps - trg_kps).pow(2).sum(dim=0).pow(0.5)
        thres = pckthres.expand_as(l2dist).float() * cls.alpha
        correct_pts = torch.le(l2dist, thres)

        correct_ids = utils.where(correct_pts == 1)
        incorrect_ids = utils.where(correct_pts == 0)
        correct_dist = l2dist[correct_pts]

        return correct_dist, correct_ids, incorrect_ids

    @classmethod
    def intersection_over_union(cls, mask1, mask2):
        r"""Computes IoU between two masks"""
        rel_part_weight = torch.sum(torch.sum(mask2.gt(0.5).float(), 2, True), 3, True) / \
                          torch.sum(mask2.gt(0.5).float())
        part_iou = torch.sum(torch.sum((mask1.gt(0.5) & mask2.gt(0.5)).float(), 2, True), 3, True) / \
                   torch.sum(torch.sum((mask1.gt(0.5) | mask2.gt(0.5)).float(), 2, True), 3, True)
        weighted_iou = torch.sum(torch.mul(rel_part_weight, part_iou)).item()

        return weighted_iou

    @classmethod
    def label_transfer_accuracy(cls, mask1, mask2):
        r"""LT-ACC measures the overlap with emphasis on the background class"""
        return torch.mean((mask1.gt(0.5) == mask2.gt(0.5)).double()).item()

    @classmethod
    def pts2ptstr(cls, pts):
        r"""Convert tensor of points to string"""
        x_str = str(list(pts[0].cpu().numpy()))
        x_str = x_str[1:len(x_str)-1]
        y_str = str(list(pts[1].cpu().numpy()))
        y_str = y_str[1:len(y_str)-1]

        return x_str, y_str

    @classmethod
    def pts2mask(cls, x_pts, y_pts, shape):
        r"""Build a binary mask tensor base on given xy-points"""
        x_idx, y_idx = draw.polygon(x_pts, y_pts, shape)
        mask = np.zeros(shape, dtype=np.bool)
        mask[x_idx, y_idx] = True

        return mask

    @classmethod
    def ptstr2mask(cls, x_str, y_str, out_h, out_w):
        r"""Convert xy-point mask (string) to tensor mask"""
        x_pts = np.fromstring(x_str, sep=',')
        y_pts = np.fromstring(y_str, sep=',')
        mask_np = cls.pts2mask(y_pts, x_pts, [out_h, out_w])
        mask = torch.tensor(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).float()

        return mask