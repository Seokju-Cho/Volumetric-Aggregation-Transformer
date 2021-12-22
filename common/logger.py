r""" Logging during training/testing """
import datetime
import logging
import os
import git

from tensorboardX import SummaryWriter
import torch


class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self, dataset):
        self.benchmark = dataset.benchmark
        self.class_ids_interest = dataset.class_ids
        self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()

        if self.benchmark == 'pascal':
            self.nclass = 20
        elif self.benchmark == 'coco':
            self.nclass = 80
        elif self.benchmark == 'fss':
            self.nclass = 1000

        self.intersection_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.union_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.ones = torch.ones_like(self.union_buf)
        self.loss_buf = []

    def update(self, inter_b, union_b, class_id, loss):
        self.intersection_buf.index_add_(1, class_id, inter_b.float())
        self.union_buf.index_add_(1, class_id, union_b.float())
        if loss is None:
            loss = torch.tensor(0.0)
        self.loss_buf.append(loss)

    def compute_iou(self):
        iou = self.intersection_buf.float() / \
              torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean() * 100

        fb_iou = (self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1) /
                  self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)).mean() * 100

        return miou, fb_iou

    def write_result(self, split, epoch):
        iou, fb_iou = self.compute_iou()

        loss_buf = torch.stack(self.loss_buf)
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch
        msg += 'Avg L: %6.5f  ' % loss_buf.mean()
        msg += 'mIoU: %5.2f   ' % iou
        msg += 'FB-IoU: %5.2f   ' % fb_iou

        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
            iou, fb_iou = self.compute_iou()
            if epoch != -1:
                loss_buf = torch.stack(self.loss_buf)
                msg += 'L: %6.5f  ' % loss_buf[-1]
                msg += 'Avg L: %6.5f  ' % loss_buf.mean()
            msg += 'mIoU: %5.2f  |  ' % iou
            msg += 'FB-IoU: %5.2f' % fb_iou
            Logger.info(msg)


class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, training, cfg=None, benchmark=None, logpath=None):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = logpath if training else '_TEST_' + args.load.split('/')[-2].split('.')[0] + logtime
        if logpath == '': logpath = logtime

        cls.logpath = os.path.join('logs', logpath)
        cls.benchmark = benchmark
        os.makedirs(cls.logpath, exist_ok=True)

        logging.basicConfig(filemode='a+',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log git commit hash
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        logging.info(f'Commit hash: {sha}')

        # Log config file
        if cfg is not None:
            logging.info('\n:=========== Few-shot Seg. with VAT ===========')
            logging.info(f'{cfg.dump()}')
            logging.info(':================================================\n')

            with open(os.path.join(cls.logpath, 'config.yaml'), 'w') as f:
                f.write(cfg.dump())

        # Log arguments
        logging.info('\n:=========== Few-shot Seg. with VAT ===========')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def load_checkpoint(cls, model, optimizer, scheduler):
        model_path = os.path.join(cls.logpath, 'model.pt')
        if not os.path.isfile(model_path):
            raise Exception('Invalid model path.')

        checkpoint = torch.load(model_path)

        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_val_miou = checkpoint['val_miou']

        return model, optimizer, scheduler, start_epoch, best_val_miou


    @classmethod
    def save_recent_model(cls, epoch, model, optimizer, scheduler, best_val_miou):
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_miou': best_val_miou,
        }, os.path.join(cls.logpath, 'model.pt'))

    @classmethod
    def save_model_miou(cls, epoch, model, optimizer, scheduler, best_val_miou):
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_miou': best_val_miou,
        }, os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. mIoU: %5.2f.\n' % (epoch, best_val_miou))

    @classmethod
    def log_params(cls, model):
        Logger.info('Backbone # param.: %d' % sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and 'backbone' in name))
        Logger.info('Learnable # param.: %d' % sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and not 'backbone' in name))
        Logger.info('Total # param.: %d' % sum(p.numel() for name, p in model.named_parameters()))

