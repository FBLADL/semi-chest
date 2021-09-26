# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from utils.model_se import Normalize, densenet121
import torch.nn.functional as F


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class Proj(nn.Module):
    def __init__(self, in_channel, hid_channel, out_channel, mlp=False, normlinear=False):
        super(Proj, self).__init__()

        self.mlp = mlp
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        linear = NormedLinear if normlinear else nn.Linear
        if mlp:
            # 3-layer MLP
            self.mlp_kept = nn.Sequential(
                linear(in_channel, hid_channel), nn.ReLU(inplace=True))
        self.mlp_drop = nn.Sequential(linear(hid_channel, hid_channel), nn.ReLU(inplace=True),
                                      linear(hid_channel, out_channel))

    def forward(self, x):
        avgpool_x = self.avgpool(x)
        avgpool_x = self.mlp_drop(self.mlp_kept(avgpool_x.squeeze())) if self.mlp else self.mlp_drop(
            avgpool_x.squeeze())

        return avgpool_x


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, dim=128, K=65536, m=0.999, T=0.07, mlp=False, joint=False, dense_local=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: utils momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.joint = joint
        self.dense_local = dense_local

        # create the encoders
        # num_classes is the output fc dimension
        backbone_q = densenet121(pretrained=True)
        backbone_k = densenet121(pretrained=True)
        feat_dim = 1024
        projector_q = Proj(feat_dim, dim, dim, mlp=mlp)
        projector_k = Proj(feat_dim, dim, dim, mlp=mlp)
        self.encoder_q = nn.Sequential(backbone_q, projector_q)
        self.encoder_k = nn.Sequential(backbone_k, projector_k)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))  # 128, 65536
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward_train(self, im_q, im_k, ratio):
        q_feat = self.encoder_q[0](im_q)

        q = self.encoder_q[1](q_feat)

        q = nn.functional.normalize(q, dim=1)
        q_feat = nn.functional.normalize(q_feat, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k_feat = self.encoder_k[0](im_k)

            k = self.encoder_q[1](k_feat)

            k = nn.functional.normalize(k, dim=1)
            k_feat = nn.functional.normalize(k_feat, dim=1)

            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits_global = torch.cat([l_pos, l_neg], dim=1)
        # labels_global = torch.zeros(
        # logits_global.shape[0], dtype=torch.long).cuda()
        logits_global /= self.T

        # self._dequeue_and_enqueue(k)

        return logits_global,  k

    def forward(self, im_q1, im_q2, im_k, ratio=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        logits1, k1 = self.forward_train(im_q1, im_k, ratio)
        logist2, k2 = self.forward_train(im_q2, im_k, ratio)

        
        self._dequeue_and_enqueue(torch.cat([k1, k2], dim=0))

        return self.forward_train(im_q1, im_k, ratio)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue2_ptr)
        assert self.K % batch_size == 0
        self.queue2[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue2_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
