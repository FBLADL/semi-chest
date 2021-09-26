import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tqdm import tqdm
from utils.dataloaders.dataloader_semi import ChestDataloader

# from utils.dataloaders.dataloader_xpert import ChexpertLoader
# from utils.dataloader import ChestDataloader
import utils.model_se as models

# from utils.model_se import FTHead
# from utils.gcloud import download_checkpoint, upload_checkpoint
import numpy as np
import math

parser = argparse.ArgumentParser(description="PyTorch Clothing1M Training")

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
# general setting
parser = argparse.ArgumentParser(description="S2MTS2 Finetuning")
parser.add_argument("--data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="densenet121",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: densenet121)",
)
parser.add_argument(
    "--epochs", default=30, type=int, metavar="N", help="number of total epochs to/run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=16,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.05,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--num-workers", default=8, type=int)
parser.add_argument(
    "--schedule",
    default=[15, 25],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument("--gpu", default=1, type=int)
parser.add_argument("--num-class", default=15, type=int)
# experiment
parser.add_argument(
    "--task", choices=["chestxray14", "chexpert", "ISIC"], nargs="*", type=str
)

# MT
parser.add_argument("--mt", action="store_true")
parser.add_argument("--cons_weight", default=1.0, type=float)
parser.add_argument("--cons_rampup", default=10, type=int)
parser.add_argument("--ema_weight", default=0.99, type=float)

# ELR
parser.add_argument("--elr", action="store_true")
parser.add_argument("--elr_weight", default=3, type=int)
parser.add_argument("--beta", default=0.9, type=float)
# semi
parser.add_argument("--label_ratio", default=2, type=int)
parser.add_argument("--runtime", default=1, type=int)

parser.add_argument("--pretrained", default="", type=str)
# options for utils v2
parser.add_argument("--mlp", action="store_true", help="use mlp head")
# docker and gcloud
parser.add_argument(
    "--gcloud", action="store_true", help="use gc cloud for storing file"
)
parser.add_argument("--dst_bucket_project", default="aiml-carneiro-research", type=str)
parser.add_argument(
    "--dst_bucket_name", default="aiml-carneiro-research-data", type=str
)
parser.add_argument("--download-name", default="s2mts2_densecl", type=str)
parser.add_argument("--upload-name", default="s2mts2", type=str)
parser.add_argument("--resize", default=256, type=int)
parser.add_argument("--user", default="fb", type=str)

args = parser.parse_args()

Labels = {
    "No Finding": 14,
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Effusion": 2,
    "Infiltration": 3,
    "Mass": 4,
    "Nodule": 5,
    "Pneumonia": 6,
    "Pneumothorax": 7,
    "Consolidation": 8,
    "Edema": 9,
    "Emphysema": 10,
    "Fibrosis": 11,
    "Pleural_Thickening": 12,
    "Hernia": 13,
}


class Moco_lincls_single(object):
    def __init__(self) -> None:
        super().__init__()
        self.create_log()

        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True

        backbone = models.__dict__[args.arch]
        self.net1, self.net2 = self.create_model(backbone, args.num_class, args.mlp)
        self.net1_ema, self.net2_ema = self.create_model(
            backbone, args.num_class, args.mlp, ema=True, imagenet=False
        )
        self.optimizer1, self.optimizer2 = self.create_optimizer()

        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.loader = ChestDataloader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_resize=args.resize,
            root_dir=args.data,
            gc_cloud=args.gcloud,
        )

        self.lr_scheulder1 = self.create_scheduler("step", self.optimizer1)
        self.lr_scheduler2 = self.create_scheduler("step", self.optimizer2)
        self.optimizer1_ema = WeightEMA(self.net1, self.net1_ema)
        self.optimizer2_ema = WeightEMA(self.net2, self.net2_ema)

    def deploy(self):
        test_loader, _ = self.loader.run(
            "test", ratio=args.label_ratio, runtime=args.runtime
        )
        labeled_loader, _ = self.loader.run(
            "labeled", ratio=args.label_ratio, runtime=args.runtime
        )
        if args.label_ratio != 100:
            unlabeled_loader, _ = self.loader.run(
                "unlabeled", ratio=args.label_ratio, runtime=args.runtime
            )
        else:
            unlabeled_loader = None
        self.warmup_criterion1 = ELR_Plus(len(labeled_loader.dataset), beta=args.beta)
        self.warmup_criterion2 = ELR_Plus(len(labeled_loader.dataset), beta=args.beta)
        # warmup_criterion = nn.MultiLabelSoftMarginLoss().cuda()

        if args.pretrained:
            prefix = os.path.join(args.user, "pretrain/s2mts2", args.download_name)
            download_checkpoint(
                args.pretrained, prefix, args.dst_bucket_project, args.dst_bucket_name
            )
            print("==> loading checkpoint {}".format(args.pretrained))
            checkpoint1 = torch.load(args.pretrained)
            state_dict = checkpoint1["state_dict"]
            for k in list(state_dict.keys()):
                # use v2mlp with three layer mlp, remove from middle
                # use v1mlp with two layer mlp, remove all
                if k.startswith("module.encoder_q"):
                    state_dict[k[len("module_encoder_q.") :]] = state_dict[k]
                elif k.startswith("module.encoder_q.mlp_kept") or k.startswith(
                    "module.encoder_q.mlp2_kept"
                ):
                    state_dict[k[len("module_encoder_q.") :]] = state_dict[k]
                del state_dict[k]
            args.start_epoch = 0
            self.net1.load_state_dict(state_dict, strict=False)
            self.net2.load_state_dict(state_dict, strict=False)
            print("==> loaded checkpoint {}".format(args.pretrained))

            # self.optimizer1_ema = WeightEMA(self.net1, self.net1_ema)
            # self.optimizer2_ema = WeightEMA(self.net2, self.net2_ema)

        for warmup in range(args.start_epoch, args.epochs):
            self.adjust_learning_rate(warmup)
            state_dict1, early_targets1 = self.pseudo_train(
                self.warmup_criterion1,
                warmup,
                self.net1,
                self.net1_ema,
                self.optimizer1,
                labeled_loader,
                unlabeled_loader,
                net_ema2=self.net2_ema,
                optimizer_ema=self.optimizer1_ema,
            )
            # self.lr_scheulder1.step(
            state_dict2, early_targets2 = self.pseudo_train(
                self.warmup_criterion2,
                warmup,
                self.net2,
                self.net2_ema,
                self.optimizer2,
                labeled_loader,
                unlabeled_loader,
                net_ema2=self.net1_ema,
                optimizer_ema=self.optimizer2_ema,
            )
            state_dict = {
                "epoch": warmup,
                "net1": state_dict1,
                "net2": state_dict2,
                "et1": early_targets1,
                "et2": early_targets2,
            }
            mean_auc = self.test(warmup, test_loader)

            checkpoint_name = "ck_epoch{}_{}".format(warmup, mean_auc * 100)
            torch.save(state_dict, checkpoint_name)

            # self.lr_scheduler2.step()

            self.save_checkpoint(filename=checkpoint_name, upload=False)
        # upload each label auc
        self.save_checkpoint(filename="result.csv")

    def save_checkpoint(self, filename="checkpoint", upload=False):
        prefix = os.path.join(args.user, "finetune", args.upload_name)
        if upload:
            upload_checkpoint(
                args.dst_bucket_project, args.dst_bucket_name, prefix, filename
            )

    def pseudo_label(self, epoch, net1, net2, net1_ema, net2_ema, optimizer1, loader_u):
        net1.eval()
        net2.eval()
        net1_ema.eval()
        net2_ema.eval()
        for batch_idx, (inputs_u_s,inputs_u_w,gt_u,item_u) in enumerate(tqdm(loader_u)):
            inputs_u_s, inputs_u_w = inputs_u_s.cuda(),inputs_u_w.cuda()
            optimiz

    def pseudo_train(
        self,
        criterion,
        epoch,
        net,
        net_ema,
        optimizer,
        loader_x,
        loader_u,
        net_ema2=None,
        optimizer_ema=None,
    ):
        net.train()
        loader_tqdm = tqdm(loader_x)
        loader = enumerate(loader_tqdm)
        net_ema.train()
        net_ema2.train()
        for batch_idx, (inputs_x_s, inputs_x_w, gt_l, item_l) in loader:
            inputs_x, labels = inputs_x_w.cuda(), gt_l.squeeze().cuda()
            optimizer.zero_grad()

            lamb = np.random.beta(1.0, 1.0)
            lamb = max(lamb, 1 - lamb)
            mix_index = torch.randperm(inputs_x.shape[0]).cuda()
            with torch.cuda.amp.autocast(enabled=True):
                outputs_x = net(inputs_x)
                outputs_x_ema = net_ema2(inputs_x).detach()

                criterion.update_hist(
                    outputs_x_ema,
                    item_l.numpy().tolist(),
                    mix_index=mix_index,
                    lamb=lamb,
                )
                loss, elr_reg = criterion(item_l, outputs_x.float(), labels.float())
                total_loss = loss + args.elr_weight * elr_reg

            early_targets = criterion.get_target()

            self.scaler.scale(total_loss).backward()
            loader_tqdm.set_description(
                f"Fine tune: loss: {loss.item()}, elr: {elr_reg}"
            )
            self.scaler.step(optimizer)
            optimizer_ema.step()
            self.scaler.update()
        # state = {'epoch': epoch, 'state_dict': net_ema.state_dict()}
        return net_ema.state_dict(), early_targets

    def train(
        self,
        criterion,
        epoch,
        net,
        net_ema,
        optimizer,
        loader_x,
        loader_u,
        net_ema2=None,
        optimizer_ema=None,
    ):
        net.train()
        loader_tqdm = tqdm(loader_x)
        loader = enumerate(loader_tqdm)
        net_ema.train()
        net_ema2.train()
        # unlabeled_iter = iter(loader_u)
        for batch_idx, (inputs_x, _, labels, item) in loader:
            # try:
            #     inputs_u = unlabeled_iter.next()
            # except:
            #     unlabeled_iter = iter(loader_u)
            #     inputs_u = unlabeled_iter.next()
            inputs_x, labels = inputs_x.cuda(), labels.squeeze().cuda()
            # inputs_u = inputs_u.cuda()
            optimizer.zero_grad()

            lamb = np.random.beta(1.0, 1.0)
            lamb = max(lamb, 1 - lamb)
            mix_index = torch.randperm(inputs_x.shape[0]).cuda()
            with torch.cuda.amp.autocast(enabled=True):
                outputs_x = net(inputs_x)
                outputs_x_ema = net_ema2(inputs_x).detach()

                # outputs_u = net(inputs_u)
                # outputs_u_ema = net_ema(inputs_u).detach()

                # outputs_u_pseudo = net_ema2(inputs_u).detach()
                # elr_targets = criterion.get_target(item)

                # mse = torch.mean((torch.sigmoid(outputs_u) -
                #                   torch.sigmoid(outputs_u_ema)) ** 2)
                # weight = 1.0 if (
                #                         epoch / args.epochs) > 1.0 else (epoch / args.epochs)

                criterion.update_hist(
                    outputs_x_ema, item.numpy().tolist(), mix_index=mix_index, lamb=lamb
                )
                loss, elr_reg = criterion(item, outputs_x.float(), labels.float())
                total_loss = loss + args.elr_weight * elr_reg

            early_targets = criterion.get_target()

            self.scaler.scale(total_loss).backward()
            loader_tqdm.set_description(
                f"Fine tune: loss: {loss.item()}, elr: {elr_reg}"
            )
            self.scaler.step(optimizer)
            optimizer_ema.step()
            self.scaler.update()
        # state = {'epoch': epoch, 'state_dict': net_ema.state_dict()}
        return net_ema.state_dict(), early_targets

    def test(self, epoch, loader):
        self.net1_ema.eval()
        self.net2_ema.eval()
        targs, preds = torch.LongTensor([]), torch.tensor([])
        with torch.no_grad():
            for batch_idx, (inputs, targets, item) in enumerate(
                tqdm(loader, desc="Test{}".format(epoch))
            ):
                inputs, targets = inputs.cuda(), targets.cuda()
                with torch.cuda.amp.autocast():
                    outputs1 = self.net1_ema(inputs)
                    outputs2 = self.net2_ema(inputs)
                outputs = (outputs1 + outputs2) / 2

                preds = torch.cat((preds, torch.sigmoid(outputs).detach().cpu()))
                targs = torch.cat((targs, targets.detach().cpu().long()))
        all_auc = [
            self.auc_roc_score(preds[:, i].squeeze(), targs.squeeze()[:, i])
            for i in range(args.num_class - 1)
        ]
        mean_auc = torch.stack(all_auc).mean()
        with open(self.log_path, "a") as f:
            f.write(
                "%03d,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f\n"
                % (
                    epoch,
                    all_auc[0],
                    all_auc[1],
                    all_auc[2],
                    all_auc[3],
                    all_auc[4],
                    all_auc[5],
                    all_auc[6],
                    all_auc[7],
                    all_auc[8],
                    all_auc[9],
                    all_auc[10],
                    all_auc[11],
                    all_auc[12],
                    all_auc[13],
                    mean_auc.item(),
                )
            )
        print("| Mean AUC {}".format(mean_auc.item()))
        return mean_auc

    def create_scheduler(self, name, optimizer):
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=10, eta_min=0.001
            )
        elif name == "step":
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 30], gamma=0.1)

    def create_log(self):
        # self.experiment_name = args.desc
        self.log_path = "result.csv"

        with open(self.log_path, "a") as f:
            f.write(
                "epoch,Atelectasis,Cardiomegaly,Effusion,Infiltration,Mass,Nodule,Pneumonia,Pneumothorax,Consolidation,Edema,Emphysema,Fibrosis,Pleural_Thickening,Hernia,Mean\n"
            )

    def create_model(self, arch, num_class, mlp, ema=False, imagenet=True):
        model1 = arch(pretrained=imagenet)
        model2 = arch(pretrained=imagenet)
        in_features = 1024
        model1.classifier = nn.Linear(in_features, num_class)
        model2.classifier = nn.Linear(in_features, num_class)
        # head1 = FTHead(in_features, num_class)
        # head2 = FTHead(in_features, num_class)
        # model1 = nn.Sequential(model1, head1)
        # model2 = nn.Sequential(model2, head2)
        model1, model2 = model1.cuda(), model2.cuda()
        # print(model1)
        if ema:
            for param in model1.parameters():
                param.detach_()
            for param in model2.parameters():
                param.detach_()
        return model1, model2

    def create_optimizer(self):
        return torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, self.net1.parameters())),
            lr=args.lr,
            betas=(0.9, 0.99),
            eps=0.1,
        ), torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, self.net2.parameters())),
            lr=args.lr,
            betas=(0.9, 0.99),
            eps=0.1,
        )

    def adjust_learning_rate(self, epoch):
        """Decay the learning rate based on schedule"""
        lr = args.lr
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
        for param_group in self.optimizer1.param_groups:
            param_group["lr"] = lr
        for param_group in self.optimizer2.param_groups:
            param_group["lr"] = lr

    def auc_roc_score(self, input, targ):
        "Computes the area under the receiver operator characteristic (ROC) curve using the trapezoid method. Restricted binary classification tasks."
        fpr, tpr = self.roc_curve(input, targ)
        d = fpr[1:] - fpr[:-1]
        sl1, sl2 = [slice(None)], [slice(None)]
        sl1[-1], sl2[-1] = slice(1, None), slice(None, -1)
        return (d * (tpr[tuple(sl1)] + tpr[tuple(sl2)]) / 2.0).sum(-1)

    def roc_curve(self, input, targ):
        "Computes the receiver operator characteristic (ROC) curve by determining the true positive ratio (TPR) and false positive ratio (FPR) for various classification thresholds. Restricted binary classification tasks."
        targ = targ == 1
        desc_score_indices = torch.flip(input.argsort(-1), [-1])
        input = input[desc_score_indices]
        targ = targ[desc_score_indices]
        d = input[1:] - input[:-1]
        distinct_value_indices = torch.nonzero(d).transpose(0, 1)[0]
        threshold_idxs = torch.cat(
            (distinct_value_indices, torch.LongTensor([len(targ) - 1]).to(targ.device))
        )
        tps = torch.cumsum(targ * 1, dim=-1)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
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
            if param.type() == "torch.cuda.LongTensor":
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
        self.pred_hist[index] = (
            self.beta * self.pred_hist[index] + (1 - self.beta) * y_pred_
        )
        self.q = (
            lamb * self.pred_hist[index] + (1 - lamb) * self.pred_hist[index][mix_index]
        )

    def get_target(self, index=None):
        if index:
            return self.pred_hist[index]
        else:
            return self.pred_hist

    def set_target(self, target):
        self.pred_hist = target


if __name__ == "__main__":
    engine = Moco_lincls_single()
    engine.deploy()
