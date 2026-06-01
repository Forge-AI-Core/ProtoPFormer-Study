# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
from cProfile import label
import math
import os
import sys
import logging
import pickle
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from torch.nn.modules.loss import _Loss
import tools.utils as utils
from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(model: torch.nn.Module, criterion: _Loss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    tb_writer: SummaryWriter, iteration: int,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    args=None,
                    set_training_mode=True,):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 30

    logger = logging.getLogger("train")
    logger.info("Start train one epoch")
    it = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs, auxi_item = model(samples)

            loss = criterion(outputs, targets)

            if args.use_ppc_loss:
                ppc_cov_coe, ppc_mean_coe = args.ppc_cov_coe, args.ppc_mean_coe
                total_proto_act, cls_attn_rollout, original_fea_len = auxi_item[2], auxi_item[3], auxi_item[4]
                if hasattr(model, 'module'):
                    ppc_cov_loss, ppc_mean_loss = model.module.get_PPC_loss(total_proto_act, cls_attn_rollout, original_fea_len, targets)
                else:
                    ppc_cov_loss, ppc_mean_loss = model.get_PPC_loss(total_proto_act, cls_attn_rollout, original_fea_len, targets)

                ppc_cov_loss = ppc_cov_coe * ppc_cov_loss
                ppc_mean_loss = ppc_mean_coe * ppc_mean_loss
                if epoch >= 20:
                    loss = loss + ppc_cov_loss + ppc_mean_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        tb_writer.add_scalars(
            main_tag="train/loss",
            tag_scalar_dict={
                "cls": loss.item(),
            },
            global_step=iteration+it
        )
        if args.use_global and args.use_ppc_loss:
            tb_writer.add_scalars(
                main_tag="train/ppc_cov_loss",
                tag_scalar_dict={
                    "ppc_cov_loss": ppc_cov_loss.item(),
                },
                global_step=iteration+it
            )
            tb_writer.add_scalars(
                main_tag="train/ppc_mean_loss",
                tag_scalar_dict={
                    "ppc_mean_loss": ppc_mean_loss.item(),
                },
                global_step=iteration+it
            )
        it += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def get_img_mask(data_loader, model, device, args):
    logger = logging.getLogger("get mask")
    logger.info("Get mask")
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Get Mask:'

    # switch to evaluation mode
    model.eval()

    all_attn_mask = []
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            cat_mask = model.get_attn_mask(images)
            all_attn_mask.append(cat_mask.cpu())
    all_attn_mask = torch.cat(all_attn_mask, dim=0) # (num, 2, 14, 14)
    if hasattr(model, 'module'):
        model.module.all_attn_mask = all_attn_mask
    else:
        model.all_attn_mask = all_attn_mask

@torch.no_grad()
def evaluate(data_loader, model, device, args):
    logger = logging.getLogger("validate")
    logger.info("Start validation")
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    cm = torch.zeros(args.nb_classes, args.nb_classes, dtype=torch.long)  # cm[정답, 예측]
    all_token_attn, pred_labels = [], []
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output, auxi_items = model(images,)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        _, pred = output.topk(k=1, dim=1)
        pred_labels.append(pred)
        for ti, pi in zip(target.detach().cpu().view(-1).tolist(),
                          pred.detach().cpu().view(-1).tolist()):
            cm[ti, pi] += 1

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        if args.use_global:
            global_acc1 = accuracy(auxi_items[2], target)[0]
            local_acc1 = accuracy(auxi_items[3], target)[0]
            metric_logger.meters['global_acc1'].update(global_acc1.item(), n=batch_size)
            metric_logger.meters['local_acc1'].update(local_acc1.item(), n=batch_size)
        all_token_attn.append(auxi_items[0])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # ── per-class 진단 (불균형 대응 확인용): confusion matrix + precision/recall/f1 + balanced_acc ──
    C = args.nb_classes
    names = getattr(args, 'class_names', None) or [str(i) for i in range(C)]
    cm_lines = ["Confusion matrix (행=정답, 열=예측):", " " * 12 + "".join(f"{n:>10}" for n in names)]
    for i in range(C):
        cm_lines.append(f"{names[i]:>12}" + "".join(f"{int(cm[i, j]):>10}" for j in range(C)))
    logger.info("\n".join(cm_lines))

    recalls = []
    rep = [f"{'class':>12}{'prec':>9}{'recall':>9}{'f1':>8}{'support':>9}"]
    for i in range(C):
        tp = int(cm[i, i]); sup = int(cm[i].sum()); pred_i = int(cm[:, i].sum())
        prec = tp / pred_i if pred_i else 0.0
        rec = tp / sup if sup else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        recalls.append(rec)
        rep.append(f"{names[i]:>12}{prec:>9.4f}{rec:>9.4f}{f1:>8.4f}{sup:>9}")
    bal_acc = sum(recalls) / len(recalls) if recalls else 0.0
    rep.append(f"balanced_acc (mean recall) = {bal_acc*100:.2f}%")
    logger.info("\n".join(rep))

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['balanced_acc'] = bal_acc * 100
    return stats