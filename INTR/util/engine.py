# ------------------------------------------------------------------------
# INTR
# Copyright (c) 2023 Imageomics Paul. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
import util.utils as utils
from timm.utils import accuracy
from .losses import DistillationLoss

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0):
    model.train()
    criterion = nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = targets.to(device)

        # filenames = [t["file_name"] for t in targets]
        # for t in targets:
        #     del t["file_name"]
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast():
            outputs, _, _, _, _ = model(samples)
            # loss = criterion(samples, outputs['query_logits'], targets)
            loss = criterion(outputs['query_logits'], targets)

        ## INTR uses only one type of loss i.e., CE loss
        # losses = sum(loss_dict[k] for k in loss_dict.keys())

        ## reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            # print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        with torch.cuda.amp.autocast():
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        torch.cuda.synchronize()
        # losses.backward()
        # if max_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # optimizer.step()


        acc1, acc5 = accuracy(outputs['query_logits'], targets, topk=(1, 5))

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(acc1=acc1)
        metric_logger.update(acc5=acc5)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device):

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = targets.to(device)
        # filenames = [t["file_name"] for t in targets]
        # for t in targets:
        #     del t["file_name"]
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast():
            outputs, _, _, _, _ = model(samples)
            loss = criterion(outputs['query_logits'], targets)

        ## reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_value = sum(loss_dict_reduced.values())
        loss_value = loss.item()

        metric_logger.update(loss=loss_value)
        # acc1, acc5, _ = utils.class_accuracy(outputs, targets, topk=(1, 5))
        acc1, acc5 = accuracy(outputs['query_logits'], targets, topk=(1, 5))
        batch_size = samples.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats