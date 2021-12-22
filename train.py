r""" VAT training (validation) code """
import argparse
import os

import torch.optim as optim
import torch.nn as nn
import torch

from config.config import get_cfg_defaults
from model.vat import VAT
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset

def train(epoch, model, dataloader, optimizer, training):
    r""" Train VAT """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        
        # 1. VAT forward pass
        batch = utils.to_cuda(batch)
        logit_mask = model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1))
        pred_mask = logit_mask.argmax(dim=1)

        # 2. Compute loss & update model parameters
        loss = model.module.compute_objective(logit_mask, batch['query_mask'])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou

def split_params(model):
    encoder_param = []
    decoder_param = []

    for name, param in model.named_parameters():
        if 'decoder' in name:
            decoder_param.append(param)
        else:
            encoder_param.append(param)

    return encoder_param, decoder_param

if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='VAT Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../Datasets_VAT')
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--load', type=str, default=None)
    args = parser.parse_args()

    
    cfg = get_cfg_defaults()
    
    if args.load is None:
        cfg.merge_from_file(args.config)
    else:
        cfg.merge_from_file(os.path.join('logs', args.load, 'config.yaml'))
        # Load from specified path
    cfg.freeze()

    logpath = args.logpath if args.load is None else args.load
    Logger.initialize(args, training=True, cfg=cfg, benchmark=cfg.TRAIN.BENCHMARK, logpath=logpath)

    # Model initialization
    model = VAT(cfg, False)
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Helper classes (for training) initialization
    encoder_param, decoder_param = split_params(model)
    optimizer = optim.AdamW([
        {"params": encoder_param, "lr": cfg.TRAIN.LR, "weight_decay": cfg.TRAIN.WEIGHT_DECAY},
        {"params": decoder_param, "lr": cfg.TRAIN.DECODER_LR, "weight_decay": cfg.TRAIN.DECODER_WEIGHT_DECAY},
    ])
    
    if cfg.TRAIN.LR_SCHEDULER == 'constant':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[] if cfg.TRAIN.MILESTONES is None else cfg.TRAIN.MILESTONES, gamma=1.)
    elif cfg.TRAIN.LR_SCHEDULER == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.NITER, eta_min=1e-6)
    else:
        raise NotImplementedError('Invalid learning rate scheduler.')
    Evaluator.initialize()

    # Dataset initialization
    FSSDataset.initialize(benchmark=cfg.TRAIN.BENCHMARK, img_size=cfg.TRAIN.IMG_SIZE, datapath=args.datapath, use_original_imgsize=False,
        apply_cats_augmentation=cfg.TRAIN.CATS_AUGMENTATIONS, apply_pfenet_augmentation=cfg.TRAIN.PFENET_AUGMENTATIONS)
    dataloader_trn = FSSDataset.build_dataloader(cfg.TRAIN.BENCHMARK, cfg.TRAIN.BSZ, cfg.SYSTEM.NUM_WORKERS, cfg.TRAIN.FOLD, 'trn')
    dataloader_val = FSSDataset.build_dataloader(cfg.TRAIN.BENCHMARK, cfg.TRAIN.BSZ, cfg.SYSTEM.NUM_WORKERS, cfg.TRAIN.FOLD, 'val')

    # Train VAT
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    start_epoch = 0

    if args.load is not None:
        model, optimizer, scheduler, start_epoch, best_val_miou =\
             Logger.load_checkpoint(model, optimizer, scheduler)

    for epoch in range(start_epoch, cfg.TRAIN.NITER):

        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True)
        scheduler.step()
        Logger.info(f'Learning rate: {scheduler.get_last_lr()}')
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False)

        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            Logger.save_model_miou(epoch, model, optimizer, scheduler, best_val_miou)
        Logger.save_recent_model(epoch, model, optimizer, scheduler, best_val_miou)

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')
