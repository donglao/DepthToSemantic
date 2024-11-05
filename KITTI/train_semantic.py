import os
import argparse
import warnings

import numpy as np
import math
import torch
import torchvision
import torchvision.transforms as transforms
import scipy.io as io

from monodepth_semantic import Monodepth2SemanticModel
from data_utils_semantic import KITTISemanticSegmentation
from train_utils import *

parser = argparse.ArgumentParser(description='Pretraining for semantic \
                                 segmentation task')
parser.add_argument('--iterations', default=20000, type=int, metavar='N',
                    help='number of iterations for finetuning')
parser.add_argument('--learningrate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--pretrain', default=None, type=str,
                    help='dataset location')
parser.add_argument('--encoder', default='resnet18', type=str,
                    help='encoder architecture')
parser.add_argument('--decoder', default='monodepth', type=str,
                    help='decoder architecture')
parser.add_argument('--datadir', default='kitti_data_semantics', type=str,
                    help='dataset location')
parser.add_argument('--trainsetsize', default=16, type=int,
                    help='how many samples are used for training')
parser.add_argument('--batchsize', default=16, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU id to use.')
parser.add_argument('--freezeencoder', dest='freeze', action='store_true',
                    help='freeze encoder and only finetune decoder')
parser.add_argument('--resultdir', default='.', type=str,
                    help='result saving directory')
parser.add_argument('--printinterval', default=100, type=int,
                    metavar='N', help='interval between printing loss')
parser.add_argument('--evaluateinterval', default=1000, type=int,
                    metavar='N', help='interval between evaluation')
args = parser.parse_args()

    
def train_semantic_seg_KITTI(args):
    # Set the training parameters
    show_parameters(args)
    
    device     = 'cuda:' + args.gpu
    lr         = args.learningrate
    max_iter   = args.iterations
    datapath   = args.datadir
    pretrain   = args.pretrain
    samplesize = args.trainsetsize
    batchsize  = args.batchsize
    encoder    = args.encoder
    decoder    = args.decoder

    if args.freeze:
        freeze_encoder = True
    else:
        freeze_encoder = False
    # Set up saving dir
    if not os.path.exists(args.resultdir):
        os.makedirs(args.resultdir)
    else:
        print('Warning: '+args.resultdir,' already exists.')  
    filename = encoder+'_'+str(pretrain)+'_Freeze_'+ \
                str(freeze_encoder)+'_trainsize_'+str(samplesize) +\
                '_lr_' + str(lr)
        
    # Set up network and data    
    net = Monodepth2SemanticModel(encoder=encoder)
    dataset = KITTISemanticSegmentation(datapath,training_samples=samplesize,
                                        reshape=[640,192])
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize,
                                               shuffle=True, num_workers=0)
    test_dataset = KITTISemanticSegmentation(datapath,training_samples=200,
                                             reshape=[640,192])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=4)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    normalize = None
    
    """
    Load pretrained models:
    depth:         encoders + decoder pretrained on depth
    depth_encoder: encoder pretrained on depth only
    ImageNet:      ImageNet pretrain only. Remark: need data normalization
    """
    if pretrain == 'depth':
        net.restore_model(encoder_restore_path=encoder+'_encoder.pth',
                          decoder_restore_path=encoder+'_decoder.pth')
    elif pretrain == 'depth_IN':
        net.restore_model(encoder_restore_path=encoder+'_encoder_IN.pth',
                          decoder_restore_path=encoder+'_decoder_IN.pth')
    elif pretrain == 'depth_full':
        net.restore_model(encoder_restore_path=encoder+'_encoder_full.pth',
                          decoder_restore_path=encoder+'_decoder_full.pth')
    elif pretrain == 'depth_encoder':
        net.restore_model(encoder_restore_path=encoder+'_encoder.pth',
                          decoder_restore_path=None)
    elif pretrain == 'flow':
        net.restore_model(encoder_restore_path=encoder+'_encoder_flow.pth',
                          decoder_restore_path=None)
    elif pretrain == 'maskcontrast':
        net.restore_model(encoder_restore_path=encoder+'_encoder_maskcontrast.pth.tar',
                          decoder_restore_path=None)
    elif pretrain == 'moco':
        net.load_MOCO_Pretrain(encoder, encoder_restore_path=encoder+'_encoder_moco.pth.tar')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    elif pretrain == 'inpainting':
        net.restore_model(encoder_restore_path=encoder+'_encoder_inpainting.pth',
                          decoder_restore_path=None)
    elif pretrain == 'nyu':
        net.restore_model(encoder_restore_path=encoder+'_encoder_nyu.pth',
                          decoder_restore_path=None)
    elif pretrain == 'ImageNet':
        net.load_ImageNet_Pretrain(encoder)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    elif pretrain is None:
        pass
    else:
        raise ValueError('Unsupported encoder pretrain...')


    if freeze_encoder is False:
        net.train()
    else:
        net.freeze_encoder()

    """
    Train Network.
    Update network for a fixed number of iterations instead of epochs.
    """
    
    net.to(device)
    iteration = 0

    # data logging, ugly, keep as-is for now
    train_loss_all = []
    test_loss_all = []
    miou_all = []
    pix_acc_all = []
    per_class_iou_all = []
    while iteration < max_iter:
        loss_total = []
        adjust_learning_rate_cosine(optimizer, iteration, max_iter, lr)
        for image, label in train_loader:
            image = image.to(device,dtype=torch.float32)
            label = label.to(device,dtype=torch.long)
            # ImageNet pretrain are done on particular data normalization
            if normalize is not None:
                image = normalize(image) 

            if iteration > max_iter:
                break
            score = net.forward(image)
            
            if False: #visualization for debugging only
                import matplotlib.pyplot as plt
                vis = np.argmax(score[0,:,:,:].cpu().detach().numpy(), axis=0)
                # vis = label[0,:,:].cpu().detach().numpy()
                plt.imshow(vis)
                plt.show()

            loss = supervised_loss(score, label, weights=None)
            loss.backward()
            optimizer.step()

            iteration += 1

            if iteration % args.printinterval == 0:
                print('Training loss: ', str(loss.item()),'; iteration: ',
                      str(iteration))
                train_loss_all.append(loss.item())

            if iteration % args.evaluateinterval == 0:
                print('Evaluating on the whole dataset...')
                # ignore warnings related to 0/0 and mean of nan
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    test_loss = test_loss_KITTI(test_loader, net, device, normalize,
                                                samplesize)
                    miou, pix_acc, per_class_iou= test_accuracy_KITTI(test_loader,
                                                  net, device, normalize, samplesize)
                print('miou: ', str(miou),'; pixel accuracy: ', str(pix_acc))
                print('Test loss: ', str(test_loss))
                test_loss_all.append(test_loss)
                miou_all.append(miou)
                pix_acc_all.append(pix_acc)
                per_class_iou_all.append(per_class_iou)
                
                # network set to eval() when testing, need to set it back
                if freeze_encoder is False:
                    net.train()
                else:
                    net.freeze_encoder()

    """
    Saving results
    Important: results saved to mat files, index shifts by 1!
    """
    io.savemat(os.path.join(args.resultdir,filename+'.mat'),{
            'train_loss'   : np.array(train_loss_all) ,
            'test_loss'    : np.array(test_loss_all) ,
            'miou'         : np.array(miou_all) ,
            'pix_acc'      : np.array(pix_acc_all) ,
            'per_class_iou': np.array(per_class_iou_all)
            })
    torch.save({'encoder': net.encoder.state_dict(),
                'decoder': net.decoder.state_dict()},
               os.path.join(args.resultdir,filename+'.pt'))

    
def test_loss_KITTI(test_loader, net, device, normalize, skip):
    net.eval()
    net.to(device)
    loss_all = []
    idx = 0
    for image, label in test_loader:
        if idx >= skip:
            image = image.to(device,dtype=torch.float32)
            if normalize is not None:
                image = normalize(image)
            
            label = label.to(device,dtype=torch.long)
            score = net.forward(image)
            loss = supervised_loss(score, label, weights=None)
            loss_all.append(loss.item())
        idx += 1
        
    return np.mean(loss_all)

def test_accuracy_KITTI(test_loader, net, device, normalize, skip):
    net.eval()
    net.to(device)
    miou_all = []
    pix_acc_all = []
    per_class_iou = []
    idx = 0
    for image, label in test_loader:
        if idx >= skip:
            image = image.to(device,dtype=torch.float32)
            if normalize is not None:
                image = normalize(image)
           
            score = net.forward(image)
            output_label = np.argmax(score.cpu().detach().numpy(),
                                     axis=1).astype(np.uint8)
            ground_truth = label.cpu().detach().numpy().astype(np.uint8)
            miou_batch, pix_acc_batch, iou_batch = get_miou_and_pix_acc(
                                                   output_label, ground_truth)
            miou_all.append(miou_batch)
            pix_acc_all.append(pix_acc_batch)
            per_class_iou.append(iou_batch)
        idx += 1
    # note: iou of classes ignored are stored as nan     
    per_class_iou = np.nanmean(np.array(per_class_iou), axis=0)
    
    return np.nanmean(miou_all), np.mean(pix_acc_all), per_class_iou

def main():
    train_semantic_seg_KITTI(args)

if __name__ == '__main__':
    main()

