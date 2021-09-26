import argparse
import builtins
import math
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from utils.model_se import densenet121
from utils.builders import builder_fa,builder
from utils.dataloaders.dataloader import ChestDataloader
from utils.dataloaders.dataloader_xpert import ChexpertLoader
from utils.dataloaders.dataloaders_isic import ISICLoader
from utils.gcloud import upload_checkpoint
from utils.gcloud import download_chexpert_unzip

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
# basic
parser = argparse.ArgumentParser(description='S2MTS2 Pretraining')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='densenet121',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: densenet121)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to/run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--num-workers', default=8, type=int)
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# DDP
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# experiment
parser.add_argument('--task', choices=['chestxray14', 'chexpert', 'isic'], nargs='*', type=str)

# Joint loss
parser.add_argument('--jcl', action='store_true')
parser.add_argument('--ratio-weight', default=2.0, type=float)
parser.add_argument('--jcl-weight', default=0.5, type=float)

# dense loss
parser.add_argument('--densecl', action='store_true')
parser.add_argument('--dense-weight', default=0.3, type=float)

# Arch config
parser.add_argument('--latent-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--mem-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--mom-m', default=0.999, type=float,
                    help='utils momentum of updating key encoder (default: 0.999)')
parser.add_argument('--t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for utils v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--cos', action='store_true')
# docker and gcloud
parser.add_argument('--gcloud', action='store_true',
                    help='use gc cloud for storing file')
parser.add_argument('--dst_bucket_project',
                    default='aiml-carneiro-research', type=str)
parser.add_argument('--dst_bucket_name',
                    default='aiml-carneiro-research-data', type=str)
parser.add_argument('--download-name', default='s2mts2', type=str)
parser.add_argument('--upload-name', default='s2mts2', type=str)
parser.add_argument('--resize', default=256, type=int)
parser.add_argument('--user', default='fb', type=str)


def main():
    args = parser.parse_args()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    # download_chexpert_unzip(args.data)
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    print("=> creating model '{}'".format(args.arch))
    backbone = densenet121
    base_encoder = create_encoder(backbone, args)
    momentum_encoder = create_encoder(backbone, args)
    model = builder_fa.MoCo(
        base_encoder, momentum_encoder,
        args.latent_dim, args.mem_k, args.mom_m, args.t).cuda(args.gpu)
    # model = builder_fa.MoCo(args.latent_dim, args.mem_k, args.mom_m, args.t, mlp=args.mlp, joint=args.jcl,
    #                               dense_local=args.densecl)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    cudnn.benchmark = True
    args.task = args.task[0]
    if args.task == 'chestxray14':
        loader = ChestDataloader(batch_size=args.batch_size, num_workers=args.num_workers, img_resize=args.resize,
                                 root_dir=args.data, gc_cloud=args.gcloud, k_crop=2)
    elif args.task == 'chexpert':
        loader = ChexpertLoader(root_path=args.data, batch_size=args.batch_size, img_resize=args.resize)
    elif args.task == 'isic':
        loader = ISICLoader(root_path=args.data, batch_size=args.batch_size, img_resize=args.resize, gcloud=True)

    pretrain_loader, train_sampler = loader.run(mode='moco_train')

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        top1, top5 = train(pretrain_loader, model, criterion, optimizer, epoch, args)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint(state={
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, args=args, ck_name='ck_{:03d}_{}.pth.tar'.format(epoch, top1.print_ck()))


def save_checkpoint(state, args, ck_name):
    torch.save(state, ck_name)
    prefix = os.path.join(args.user, 'pretrain/s2mts2', args.upload_name)
    upload_checkpoint(args.dst_bucket_project, args.dst_bucket_name, prefix, ck_name)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    jcl_losses = AverageMeter('jcl_Loss', ':.4e')
    cls_losses = AverageMeter('CLS_Loss', ':.4e')
    print_infos = [batch_time, data_time, losses,
                   jcl_losses, cls_losses, top1, top5]
    progress = ProgressMeter(
        len(train_loader),
        print_infos,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    end = time.time()
    num_iter = int(len(train_loader.dataset) / train_loader.batch_size)
    for i, (images, _, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            if args.jcl:
                images = [m.cuda(args.gpu, non_blocking=True) for m in images]
                images[1] = torch.cat(images[1:], dim=0)
            else:
                images[0] = images[0].cuda(args.gpu, non_blocking=True)
                images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            ratio = args.ratio_weight * ((epoch + 1) / args.epochs) if args.jcl else None
            output, target,jcl_loss = model(im_q=images[0], im_k=images[1],
                                             ratio=ratio)
            cls_loss = criterion(output, target)
            final_loss = cls_loss + jcl_loss

        jcl_losses.update(jcl_loss.item(), output.size(0))
        cls_losses.update(cls_loss.item(), output.size(0))
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        # TODO acc global and dense
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(final_loss.item(), images[0].size(0))
        #
        # jcl_losses.update(jcl_loss.item(), output[0].shape[0]) if args.jcl else None
        # dense_losses.update(loss_dense.item(), images[0].shape[0]) if args.densecl else None
        # global_losses.update(loss_global.item(), images[0].shape[0])

        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        scaler.scale(final_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return top1, top5


def create_encoder(arch, args):
    model = arch(pretrained=True)
    in_features = model.classifier.in_features
    if args.mlp:
        model.classifier = nn.Linear(in_features, in_features)
        # model.classifier_mlp3 = nn.Linear(in_features, args.moco_dim, bias=False)
        model.classifier_group = nn.Sequential(nn.Linear(in_features, in_features), nn.ReLU(
            inplace=True), nn.Linear(in_features, args.latent_dim))
    else:
        model.classifier = nn.Linear(in_features, args.latent_dim)
    return model


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def print_ck(self):
        return self.avg


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
