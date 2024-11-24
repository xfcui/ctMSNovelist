import sys

import numpy as np
from torch import nn
from tqdm import tqdm
import torch

from multi_train_utils.distributed_utils import reduce_value, is_main_process


def train_one_epoch(model, optimizer, data_loader, cv_sampler, device, epoch, args):
    model.train()
    # loss_function = torch.nn.CrossEntropyLoss()
    criterion_sm = nn.CrossEntropyLoss()
    criterion_H = nn.MSELoss()
    loss_weight_sm = args.loss_weight_sm
    loss_weight_H = args.loss_weight_H
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)
    loss_ls = []
    for step, batch in enumerate(data_loader):
        fp_, mf, gt, lab, smiles_lab = batch
        fp, _ = cv_sampler.sample_(fp_)

        fp = fp.cuda()
        mf = mf.cuda()
        gt = gt.cuda()
        lab = lab.cuda()
        H_num = mf[:, -1].reshape(-1, 1).cuda()
        inputs = {"FP": fp, "MF": mf, "tokens_X": gt}

        optimizer.zero_grad()
        y_pred, estimated_h_sum = model(inputs)

        # pred = model_msnovelist(images.to(device))
        loss_sm = criterion_sm(y_pred, lab)
        loss_H = criterion_H(estimated_h_sum.float(), H_num)
        loss = loss_weight_sm * loss_sm + loss_weight_H * loss_H
        loss.backward()

        loss = reduce_value(loss, average=True)
        loss_ls.append(loss.detach().item())
        # mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return np.mean(loss_ls)


@torch.no_grad()
def start_validate(model, data_loader, cv_sampler, device, which="val"):
    model.eval()
    criterion_sm = nn.CrossEntropyLoss()

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    loss_ls = []
    for step, data in enumerate(data_loader):
        fp_, mf, gt, lab, smiles_lab = data
        if which == "val":
            fp_, _ = cv_sampler.sample_(fp_)
        fp = fp_.cuda()
        mf = mf.cuda()
        gt = gt.cuda()
        lab = lab.cuda()
        inputs = {"FP": fp, "MF": mf, "tokens_X": gt}

        y_pred, estimated_h_sum = model(inputs)
        loss = criterion_sm(y_pred.transpose(1, 2).float(), lab.long())
        loss_ls.append(loss.item())

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return np.mean(loss_ls)
