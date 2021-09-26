import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tqdm import tqdm
from typing_extensions import runtime
import math
from utils.dataloaders.dataloaders_isic import ISICLoader
# from moco.dataloader import ChestDataloader
import utils.model_se as models
from utils.gcloud import download_checkpoint, upload_checkpoint

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
# general setting
parser.add_argument('--batch-size', default=16,
                    type=int, help='train batchsize')
parser.add_argument('-a', '--arch', metavar='ARCH', default='densenet121',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: densenet121)')

parser.add_argument('--lr', '--learning_rate', default=0.05,
                    type=float, help='initial learning rate')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--num-workers', default=8, type=int)
parser.add_argument('--data', default='chestxray/',
                    type=str, help='path to dataset')
parser.add_argument('--num-class', default=7, type=int)
parser.add_argument('--save', default='multi_label', type=str)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--desc', default='baseline',
                    help='description of experiment')
parser.add_argument('--mlp', action='store_true',
                    help='replace linear with MLP')
parser.add_argument('--gpu', default=0, type=int)
# warmup
parser.add_argument('--warmup-epochs', default=80, type=int)
parser.add_argument('--pretrained', default='', type=str)

# DGX
parser.add_argument('--gcloud', action='store_true',
                    help='use gc cloud for storing file')
parser.add_argument('--user', default='fb', type=str)
parser.add_argument('--dst_bucket_project',
                    default='aiml-carneiro-research', type=str)
parser.add_argument('--dst_bucket_name',
                    default='aiml-carneiro-research-data', type=str)
parser.add_argument('--upload-name', default='baseline_moco', type=str)
parser.add_argument('--v2mlp', action='store_true', help='use simclr v2 mlp')
parser.add_argument('--resize', default=512, type=int)
parser.add_argument('--imagenet', action='store_true')
parser.add_argument('--download-name', default='fa_121', type=str)
parser.add_argument('--freeze-net', action='store_true',
                    help='freeze feature layers')
parser.add_argument('--fine-tune-first', action='store_true',
                    help='fine tune from first mlp head or from middle mlp')
parser.add_argument('--ratio', default=2, type=int)
parser.add_argument('--runtime', default=1, type=int)
args = parser.parse_args()


class Moco_lincls_single(object):
    def __init__(self) -> None:
        super().__init__()
        self.create_log()

        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True

        backbone = models.__dict__[args.arch]
        self.net1 = self.create_model(
            backbone, args.num_class, args.mlp, False)
        self.net1_ema = self.create_model(
            backbone, args.num_class, args.mlp, True)

        self.optimizer1 = self.create_optimizer()

        self.optimizer1_ema = WeightEMA(self.net1, self.net1_ema)
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.loader = ISICLoader(
            root_path=args.data, batch_size=args.batch_size, img_resize=args.resize, gcloud=False)

        self.lr_scheulder1 = self.create_scheduler('step')

    def deploy(self):
        test_loader, _ = self.loader.run('test')
        labeled_loader, _ = self.loader.run(
            'labeled', ratio=20, runtime=args.runtime)
        unlabeled_loader, _ = self.loader.run(
            'unlabeled', ratio=20, runtime=args.runtime)
        # if args.ratio != 100:
        #     unlabeled_loader, _ = self.loader.run(
        #         'unlabeled', )
        # else:
        #     unlabeled_loader = None
        warmup_criterion = nn.CrossEntropyLoss().cuda()

        warmup = 0
        if args.pretrained:
            prefix = os.path.join(args.user, args.download_name, 'checkpoints')
            # download_checkpoint(args.pretrained, prefix,
            #                     args.dst_bucket_project, args.dst_bucket_name)
            print('==> loading checkpoint {}'.format(args.pretrained))
            checkpoint1 = torch.load(args.pretrained)
            state_dict = checkpoint1['state_dict']
            for k in list(state_dict.keys()):
                # use v2mlp with three layer mlp, remove from middle
                # use v1mlp with two layer mlp, remove all
                if k.startswith('module.encoder_q') and not k.startswith(
                        'module.encoder_q.{}'.format('classifier_group' if args.v2mlp else 'classifier')):
                    state_dict[k[len('module.encoder_q.'):]] = state_dict[k]
                del state_dict[k]
            args.start_epoch = 0
            self.net1.load_state_dict(state_dict, strict=False)
            print('==> loaded checkpoint {}'.format(args.pretrained))
        # self.optimizer1_ema = WeightEMA(self.net1, self.net1_ema)
        if args.freeze_net:
            for name, param in self.net1.named_parameters():
                if name not in ['classifier_group.weight', 'classifier_group.bias']:
                    param.requires_grad = False
            self.net1.classifier_group.weight.data.normal_(mean=0.0, std=0.01)
            self.net1.classifier_group.bias.data.zero_()

        for warmup in range(args.start_epoch, args.warmup_epochs):
            self.adjust_learning_rate(warmup)
            state_dict = self.warmup_train(warmup_criterion, warmup, self.net1, self.net1_ema,
                                           self.optimizer1, self.optimizer1_ema, labeled_loader, unlabeled_loader, 1)
            # self.lr_scheulder1.step()
            mean_auc = self.test(warmup, test_loader)

            checkpoint_name = 'net{}_epoch{}_{}'.format(
                1, warmup, mean_auc * 100)
            torch.save(state_dict, checkpoint_name)

            self.save_checkpoint(filename=checkpoint_name)
        # upload each label auc
        self.save_checkpoint(filename='result.csv')

    def save_checkpoint(self, filename='checkpoint'):
        prefix = os.path.join(args.user, args.upload_name, 'checkpoints')
        upload_checkpoint(args.dst_bucket_project,
                          args.dst_bucket_name, prefix, filename)

    def warmup_train(self, criterion, epoch, net, net_ema, optimizer, optimizer1_ema, labeled_loader, unlabeled_loader,
                     train_flag):
        # net.eval()
        net.train()
        net_ema.train()
        unlabeled_iter = iter(unlabeled_loader)

        loader = enumerate(
            tqdm(labeled_loader, desc='Warmup {}'.format(epoch)))
        for batch_idx, (inputs, labels, _) in loader:
            try:
                inputs_u, _, _ = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_loader)
                inputs_u, _, _ = unlabeled_iter.next()
            inputs, labels = inputs[0].cuda(
            ), labels.argmax(dim=1).cuda()
            inputs_u = inputs_u[0].cuda()

            with torch.cuda.amp.autocast(enabled=True):
                outputs_x = net(inputs)

                outputs_u = net(inputs_u)
                outputs_u_ema = net_ema(inputs_u).detach()

                mse = torch.mean(
                    (torch.softmax(outputs_u, dim=1) - torch.softmax(outputs_u_ema, dim=1)) ** 2)
                weight = 1.0 if (
                    epoch / args.epochs) > 1.0 else (epoch / args.epochs)
                loss = criterion(outputs_x.float(),
                                 labels.long()) + weight * mse
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            optimizer1_ema.step()
            self.scaler.update()
            optimizer.zero_grad()
            # print('loss {}'.format(loss.item()))
        state = {'epoch': epoch, 'state_dict': net.state_dict(),
                 'optimizer': optimizer.state_dict()}
        return state
        # torch.save(state, os.path.join(
        #     self.net_path, 'net{}_epoch_{}.pth.tar'.format(train_flag, epoch)))

    def adjust_learning_rate(self, epoch):
        """Decay the learning rate based on schedule"""
        lr = args.lr
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        for param_group in self.optimizer1.param_groups:
            param_group['lr'] = lr

    def test(self, epoch, loader):
        self.net1_ema.eval()
        targs, preds = torch.LongTensor([]), torch.tensor([])
        with torch.no_grad():
            for batch_idx, (inputs, targets, _) in enumerate(tqdm(loader, desc='Test{}'.format(epoch))):
                inputs, targets = inputs[0].cuda(), targets.cuda()
                with torch.cuda.amp.autocast():
                    outputs1 = self.net1_ema(inputs)
                # outputs = (outputs1 + outputs2) / 2
                outputs = outputs1

                preds = torch.cat(
                    (preds, torch.softmax(outputs, dim=1).detach().cpu()))
                targs = torch.cat((targs, targets.detach().cpu().long()))
        print(preds.shape)
        print(targs.shape)
        all_auc = [self.auc_roc_score(preds[:, i].squeeze(), targs.squeeze()[:, i])
                   for i in range(7)]
        mean_auc = torch.stack(all_auc).mean()
        # print(mean_auc)
        # with open(self.log_path, 'a') as f:
        #     f.write(
        #         '%03d,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f\n' % (
        #             epoch, all_auc[0], all_auc[1], all_auc[2], all_auc[3], all_auc[4], all_auc[5], all_auc[6],
        #             all_auc[7], all_auc[8], all_auc[9], all_auc[10], all_auc[11], all_auc[12], all_auc[13],
        #             mean_auc.item()
        #         ))
        print('| Mean AUC {}'.format(mean_auc.item()))
        return mean_auc

    def create_scheduler(self, name):
        if name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer1, T_max=10, eta_min=0.001)
        elif name == 'step':
            return torch.optim.lr_scheduler.MultiStepLR(self.optimizer1, [15, 25],
                                                        gamma=0.1)

    def create_log(self):
        self.experiment_name = args.desc
        # self.img_path = os.path.join(args.save, self.experiment_name, 'img')
        # self.net_path = os.path.join(args.save, self.experiment_name, 'net')
        self.log_path = 'result.csv'
        # self.log_path = os.path.join(
        #     args.save, self.experiment_name, 'results.csv')

        with open(self.log_path, 'a') as f:
            f.write(
                'epoch,Atelectasis,Cardiomegaly,Effusion,Infiltration,Mass,Nodule,Pneumonia,Pneumothorax,Consolidation,Edema,Emphysema,Fibrosis,Pleural_Thickening,Hernia,Mean\n')

    def create_model(self, arch, num_class, mlp, ema):
        model1 = arch(pretrained=True, progress=True)
        in_features = model1.classifier.in_features
        model1.classifier = nn.Linear(in_features, in_features)
        model1.classifier_group = nn.Linear(in_features, num_class)
        # model1.classifier_group = nn.Sequential(nn.Linear(in_features, in_features), nn.LeakyReLU(0.1, inplace=True),
        #                                         nn.Linear(in_features, num_class, bias=False))
        if ema:
            for param in model1.parameters():
                param.detach_()
        return model1.cuda()

    def create_optimizer(self):
        parameters = list(
            filter(lambda p: p.requires_grad, self.net1.parameters()))
        return torch.optim.Adam(parameters,
                                lr=args.lr,
                                betas=(0.9, 0.99),
                                eps=0.1)

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


if __name__ == '__main__':
    engine = Moco_lincls_single()
    engine.deploy()
