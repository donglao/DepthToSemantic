import os
import sys
import time
import math

import torch
import torchvision
import torch.nn.functional as F
import numpy as np

def supervised_loss(score, label, weights=None):
    # ignore index = 0: class 0 corresponds to 'unlabeled'
    loss_fn_ = torch.nn.NLLLoss(weight=weights, ignore_index=0)
    loss = loss_fn_(F.log_softmax(score, dim=1), label)
    return loss

def adjust_learning_rate_cosine(optimizer, epoch, max_epoch, initial_lr):
    """Decay the learning rate based on schedule"""
    lr = initial_lr
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / max_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # print('Learning rate adjusted to '+str(lr))

def show_parameters(args):
    print('Encoder type: ', args.encoder)
    print('Pretrain method: ', args.pretrain)
    print('Batch size: ', str(args.batchsize))
    print('Training set size: ', str(args.trainsetsize))
    print('Initial learning rate: ', str(args.learningrate))
    print('Max iterations: ', str(args.iterations))

def get_miou_and_pix_acc(score, label,num_channels=34):
    ignore_list = [-1,0,1,2,3,4,5,6,9,10,12,13,14,15,16,17,18,19,20,24,25,27,28,29,30,31,32]
    # batch * h * w
    intersection = (score==label)
    acc_map = intersection * score
    iou = np.ones(num_channels)*np.nan
    with np.errstate(divide='ignore'):
        for i in range(1,num_channels): #important! ignore category 0 which is [unlabeled]
            tp = np.sum(acc_map==i, axis = (1,2))
            p_score = np.sum(score==i, axis = (1,2))
            p_label = np.sum(label==i, axis = (1,2))
            iou[i] = np.nanmean(np.divide(tp,p_score+p_label-tp)) #np.divide: return nan when 0/0

        iou[ignore_list] = np.nan #remove unwanted classes            
        miou = np.nanmean(iou)
        pix_acc = np.sum(intersection)/np.sum(label>0) #label>0: classes of interest
        
    return miou , pix_acc, iou
