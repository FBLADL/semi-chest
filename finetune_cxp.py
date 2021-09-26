import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tqdm import tqdm
from utils.dataloaders.dataloader_semi import ChestDataloader
from utils.dataloaders.dataloader_xpert import ChexpertLoader
# from utils.dataloader import ChestDataloader
import utils.model_se as models
# from utils.model_se import FTHead
from utils.gcloud import download_checkpoint, upload_checkpoint
import numpy as np
import math

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
# general setting
parser = argparse.ArgumentParser(description='S2MTS2 Finetuning')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='densenet121',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: densenet121)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to/run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--num-workers', default=8, type=int)
parser.add_argument('--schedule', default=[15, 25], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu', default=1, type=int)
parser.add_argument('--num-class', default=14, type=int)
# experiment
parser.add_argument(
    '--task', choices=['chestxray14', 'chexpert', 'isic'], nargs='*', type=str)

# MT
parser.add_argument('--mt', action='store_true')
parser.add_argument('--cons_weight', default=1.0, type=float)
parser.add_argument('--cons_rampup', default=10, type=int)
parser.add_argument('--ema_weight', default=0.99, type=float)

# ELR
parser.add_argument('--elr', action='store_true')
parser.add_argument('--elr_weight', default=3, type=int)
parser.add_argument('--beta', default=0.9, type=float)
# semi
parser.add_argument('--ratio', default=2, type=int)
parser.add_argument('--runtime', default=1, type=int)

parser.add_argument('--pretrained', default='', type=str)
# options for utils v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
# docker and gcloud
parser.add_argument('--gcloud', action='store_true',
                    help='use gc cloud for storing file')
parser.add_argument('--dst_bucket_project',
                    default='aiml-carneiro-research', type=str)
parser.add_argument('--dst_bucket_name',
                    default='aiml-carneiro-research-data', type=str)
parser.add_argument('--download-name', default='s2mts2_densecl', type=str)
parser.add_argument('--upload-name', default='s2mts2', type=str)
parser.add_argument('--resize', default=224, type=int)
parser.add_argument('--user', default='fb', type=str)
parser.add_argument('--policy', default='zeros', type=str)

args = parser.parse_args()

# Labels = {'No Finding': 14, 'Atelectasis': 0, 'Cardiomegaly': 1, 'Effusion': 2, 'Infiltration': 3, 'Mass': 4,
#           'Nodule': 5, 'Pneumonia': 6, 'Pneumothorax': 7,
#           'Consolidation': 8, 'Edema': 9, 'Emphysema': 10, 'Fibrosis': 11, 'Pleural_Thickening': 12,
#           'Hernia': 13}
Labels = {'No Finding': 0, 'Enlarged Cardiomediastinum': 1, 'Cardiomegaly': 2, 'Lung Opacity': 3, 'Lung Lesion': 4,
          'Edema': 5, 'Consolidation': 6, 'Pneumonia': 7, 'Atelectasis': 8, 'Pneumothorax': 9, 'Pleural Effusion': 10,
          'Pleural Other': 11, 'Fracture': 12, 'Support Devices': 13}

Reverse_Labels = {v: k for k, v in Labels.items()}


class Moco_lincls_single(object):
    def __init__(self) -> None:
        super().__init__()
        self.create_log()

        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True

        backbone = models.__dict__[args.arch]
        self.net1 = self.create_model(
            backbone, args.num_class, args.mlp, ema=False)
        self.net1_ema = self.create_model(
            backbone, args.num_class, args.mlp, ema=True)
        self.optimizer1 = self.create_optimizer()

        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.loader = ChexpertLoader(
            root_path=args.data, batch_size=args.batch_size, img_resize=args.resize)
        # self.loader = ChestDataloader(
        #     batch_size=args.batch_size, num_workers=args.num_workers, img_resize=args.resize, root_dir=args.data,
        #     gc_cloud=args.gcloud)

        # self.lr_scheduler2 = self.create_scheduler('step', self.optimizer2)
        # self.optimizer1_ema = WeightEMA(self.net1, self.net1_ema)
        # self.optimizer2_ema = WeightEMA(self.net2, self.net2_ema)

    def deploy(self):
        test_loader, _ = self.loader.run(
            'test', ratio=args.ratio)
        labeled_loader, _ = self.loader.run(
            'label', ratio=args.ratio)
        unlabeled_loader, _ = self.loader.run('unlabel', ratio=args.ratio)
        warmup_criterion = nn.MultiLabelSoftMarginLoss().cuda()
        # warmup_criterion = nn.BCEWithLogitsLoss().cuda()

        if args.pretrained:
            prefix = os.path.join(
                args.user, 'pretrain/s2mts2', args.download_name)
            # download_checkpoint(args.pretrained, prefix,
            #                     args.dst_bucket_project, args.dst_bucket_name)
            print('==> loading checkpoint {}'.format(args.pretrained))
            checkpoint1 = torch.load(args.pretrained)
            state_dict = checkpoint1['state_dict']
            for k in list(state_dict.keys()):
                # use v2mlp with three layer mlp, remove from middle
                # use v1mlp with two layer mlp, remove all
                if k.startswith('module.encoder_q') and not k.startswith(
                        'module.encoder_q.{}'.format('classifier_group' if args.mlp else 'classifier')):
                    state_dict[k[len('module.encoder_q.'):]] = state_dict[k]
                del state_dict[k]
            args.start_epoch = 0
            self.net1.load_state_dict(state_dict, strict=False)
            # self.net2.load_state_dict(state_dict, strict=False)
            print('==> loaded checkpoint {}'.format(args.pretrained))
        self.optimizer1_ema = WeightEMA(self.net1, self.net1_ema)
        for warmup in range(args.start_epoch, args.epochs):
            self.adjust_learning_rate(warmup)
            state_dict1 = self.train(warmup_criterion, warmup, self.net1, self.net1_ema,
                                     self.optimizer1,
                                     labeled_loader, unlabeled_loader,
                                     self.optimizer1_ema)
            # self.lr_scheulder1.step()
            state_dict = {'epoch': warmup,
                          'net1': state_dict1, }
            mean14_auc = self.test(warmup, test_loader)
            checkpoint_name = 'ck_epoch{}_{}'.format(
                warmup, mean14_auc * 100)
            torch.save(state_dict, checkpoint_name)

            # self.lr_scheduler2.step()

            self.save_checkpoint(filename=checkpoint_name, upload=False)
        # upload each label auc
        self.save_checkpoint(filename='result.csv')

    def save_checkpoint(self, filename='checkpoint', upload=False):
        prefix = os.path.join(args.user, 'finetune', args.upload_name)
        if upload:
            upload_checkpoint(args.dst_bucket_project,
                              args.dst_bucket_name, prefix, filename)

    def train(self, criterion, epoch, net, net_ema, optimizer, loader_x, loader_u, optimizer_ema=None):
        net.train()
        loader_tqdm = tqdm(loader_x)
        loader = enumerate(loader_tqdm)
        net_ema.train()
        unlabeled_iter = iter(loader_u)

        for batch_idx, (inputs_x, labels, item) in loader:
            try:
                inputs_u, _, _ = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(loader_u)
                inputs_u, _, _ = unlabeled_iter.next()
            inputs_x, labels = inputs_x[0].cuda(), labels.squeeze().cuda()
            inputs_u = inputs_u[0].cuda()
            # optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=True):
                outputs_x = net(inputs_x)

                outputs_u = net(inputs_u)
                outputs_u_ema = net_ema(inputs_u).detach()

                mse = torch.mean(
                    (torch.sigmoid(outputs_u) - torch.sigmoid(outputs_u_ema)) ** 2)
                weight = 1.0 if (
                    epoch / args.epochs) > 1.0 else (epoch / args.epochs)

                # outputs_u_pseudo = net_ema2(inputs_u).detach()

                loss = criterion(
                    outputs_x.float(), labels.float()) + weight * mse 

            # early_targets = criterion.get_target()

            self.scaler.scale(loss).backward()
            loader_tqdm.set_description(
                f'Fine tune: loss: {loss.item()}')
            self.scaler.step(optimizer)
            optimizer_ema.step()
            self.scaler.update()
            optimizer.zero_grad()
        state = {'epoch': epoch, 'state_dict': net_ema.state_dict()}
        return state

    def test(self, epoch, loader):
        self.net1_ema.eval()
        # self.net1_ema.eval()
        # self.net2_ema.eval()
        targs, preds = torch.LongTensor([]), torch.tensor([])
        with torch.no_grad():
            for batch_idx, (inputs, targets, item) in enumerate(tqdm(loader, desc='Test{}'.format(epoch))):
                inputs, targets = inputs[0].cuda(), targets.cuda()
                with torch.cuda.amp.autocast():
                    outputs1 = self.net1_ema(inputs)
                    # outputs2 = self.net2_ema(inputs)
                outputs = outputs1

                preds = torch.cat(
                    (preds, torch.sigmoid(outputs).detach().cpu()))
                targs = torch.cat((targs, targets.detach().cpu().long()))
        all_auc = [self.auc_roc_score(preds[:, i].squeeze(), targs.squeeze()[:, i])
                   for i in range(args.num_class)]
        mean14_auc = torch.stack(all_auc).mean()
        # mean5_auc = torch.stack([all_auc[2], all_auc[5], all_auc[6], all_auc[8], all_auc[10]]).mean()
        # mean_auc = torch.stack(all_auc).mean()
        with open(self.log_path, 'a') as f:
            f.write(
                '%03d,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f\n' % (
                    epoch, all_auc[0], all_auc[1], all_auc[2], all_auc[3], all_auc[4], all_auc[5], all_auc[6],
                    all_auc[7], all_auc[8], all_auc[9], all_auc[10], all_auc[11], all_auc[12], all_auc[13],
                    mean14_auc.item()
                ))
        print('| Mean 14 AUC {}'.format(mean14_auc.item()))
        return mean14_auc

    def create_scheduler(self, name, optimizer):
        if name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)
        elif name == 'step':
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 30],
                                                        gamma=0.1)

    def create_log(self):
        # self.experiment_name = args.desc
        self.log_path = 'result.csv'

        with open(self.log_path, 'a') as f:
            f.write(
                f'epoch,{Reverse_Labels[0]},{Reverse_Labels[1]},{Reverse_Labels[2]},{Reverse_Labels[3]},{Reverse_Labels[4]},{Reverse_Labels[5]},{Reverse_Labels[6]},{Reverse_Labels[7]},{Reverse_Labels[8]},{Reverse_Labels[9]},{Reverse_Labels[10]},{Reverse_Labels[11]},{Reverse_Labels[12]},{Reverse_Labels[13]}, Mean14, Mean5\n')

    def create_model(self, arch, num_class, mlp, ema):
        model1 = arch(pretrained=True, progress=True)
        in_features = model1.classifier.in_features
        model1.classifier = nn.Linear(in_features, in_features)
        # model1.classifier_group = nn.Linear(in_features, num_class)
        model1.classifier_group = nn.Sequential(nn.Linear(in_features, in_features), nn.LeakyReLU(0.1, inplace=True),
                                                nn.Linear(in_features, num_class))
        if ema:
            for param in model1.parameters():
                param.detach_()
        return model1.cuda()

    def create_optimizer(self):
        return torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.net1.parameters())), 
                                 lr=args.lr,
                                 betas=(0.9, 0.99),
                                 eps=0.1)

    def adjust_learning_rate(self, epoch):
        """Decay the learning rate based on schedule"""
        lr = args.lr
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        for param_group in self.optimizer1.param_groups:
            param_group['lr'] = lr

    def auc_roc_score(self, input, targ):
        "Computes the area under the receiver operator characteristic (ROC) curve using the trapezoid method. Restricted binary classification tasks."
        fpr, tpr = self.roc_curve(input, targ)
        d = fpr[1:] - fpr[:-1]
        sl1, sl2 = [slice(None)], [slice(None)]
        sl1[-1], sl2[-1] = slice(1, None), slice(None, -1)
        return (d * (tpr[tuple(sl1)] + tpr[tuple(sl2)]) / 2.).sum(-1)

    def roc_curve(self, input, targ):
        "Computes the receiver operator characteristic (ROC) curve by determining the true positive ratio (TPR) and false positive ratio (FPR) for various classification thresholds. Restricted binary classification tasks."
        targ = (targ == 1)
        desc_score_indices = torch.flip(input.argsort(-1), [-1])
        input = input[desc_score_indices]
        targ = targ[desc_score_indices]
        d = input[1:] - input[:-1]
        distinct_value_indices = torch.nonzero(d).transpose(0, 1)[0]
        threshold_idxs = torch.cat(
            (distinct_value_indices, torch.LongTensor([len(targ) - 1]).to(targ.device)))
        tps = torch.cumsum(targ * 1, dim=-1)[threshold_idxs]
        fps = (1 + threshold_idxs - tps)
        if tps[0] != 0 or fps[0] != 0:
            zer = fps.new_zeros(1)
            fps = torch.cat((zer, fps))
            tps = torch.cat((zer, tps))
        fpr, tpr = fps.float() / fps[-1], tps.float() / tps[-1]
        return fpr, tpr


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        # self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            # fix the error 'RuntimeError: result type Float can't be cast to the desired output type Long'
            # print(param.type())
            if param.type() == 'torch.cuda.LongTensor':
                ema_param = param
            else:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            # param.mul_(1 - self.wd)


class ELR_Plus(nn.Module):
    def __init__(self, num_examp, num_classes=15, beta=0.7):
        super().__init__()
        self.pred_hist = torch.zeros(num_examp, num_classes).cuda()
        self.beta = beta
        self.base_loss = nn.MultiLabelSoftMarginLoss().cuda()

    def forward(self, index, output, label):
        y_pred = torch.sigmoid(output)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)

        # loss = torch.mean(-torch.sum(label *
        #                              F.log_softmax(output, dim=1), dim=-1))
        loss = self.base_loss(output, label)
        reg = ((1 - (self.q * y_pred)).log()).mean()
        return loss, reg

    def update_hist(self, out, index, mix_index, lamb):
        y_pred_ = torch.sigmoid(out).detach()
        self.pred_hist[index] = self.beta * \
            self.pred_hist[index] + (1 - self.beta) * y_pred_
        self.q = lamb * self.pred_hist[index] + \
            (1 - lamb) * self.pred_hist[index][mix_index]

    def get_target(self, index=None):
        if index:
            return self.pred_hist[index]
        else:
            return self.pred_hist

    def set_target(self, target):
        self.pred_hist = target


if __name__ == '__main__':
    engine = Moco_lincls_single()
    engine.deploy()
