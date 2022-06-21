#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 13:48:40 2019

@author: ee18d001
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:18:40 2019

@author: ee18d001
"""
from __future__ import absolute_import
import __init_paths

#import epdb
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from pysot.core.config import cfg
from torch.autograd  import Variable
from pysot.datasets.dataset_draft import TrkDataset
from torch.utils.data import DataLoader
from pysot.models.model_builder import ModelBuilder
from pysot.models.block_a1 import _Block_A,MyBridge,fn ########################
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
#from bridge import MyBridge

##########
#import autograd.numpy as np
#from autograd import grad
#from autograd import elementwise_grad as egrad
##########
########################################################################
my_block=_Block_A.apply
my_bridge=MyBridge.apply
class SiamNet_D(nn.Module):
    def __init__(self):
        super(SiamNet_D, self).__init__()
#        self.config = Config()
        self.nc = 3
        self.discriminator = nn.Sequential(
#            OrderedDict([
	            # 3x127x127 - (3,64,3,2)
#	            ('conv1', nn.Conv2d(self.nc, 64, 3, 2, bias=True) ),
#	            ('bn1',	nn.BatchNorm2d(64) ),
#	            ('lrelu1', nn.LeakyReLU(0.01,inplace=True) ),
#
#                # 64x63x63 - (64,64,3,2)
#                ('pool1', nn.MaxPool2d(3, 2)),
#	            
#	            # 64x31x31 - (64,128,3,2)
#	            ('conv2', nn.Conv2d(64, 128, 3, 2, bias=True) ),
#	            ('bn2',	nn.BatchNorm2d(128) ),
#	            ('lrelu2', nn.LeakyReLU(0.01, inplace=True) ),
#
#                # 128x15x15 - (128,128,3,2)
#                ('pool2', nn.MaxPool2d(3, 2)),
#	            
#	            # 128x7x7 - (128,256,3,1)
#	            ('conv3', nn.Conv2d(128, 256, 3, 1, bias=True) ),
#	            ('bn3',	nn.BatchNorm2d(256) ),
#	            ('lrelu3', nn.LeakyReLU(0.01,inplace=True) ),
#	            
#	            # 256x5x5 - (256,512,3,1)
#	            ('conv4', nn.Conv2d(256, 512, 3, 1, bias=True) ),
#	            ('bn4',	nn.BatchNorm2d(512) ),
#	            ('lrelu4', nn.LeakyReLU(0.01,inplace=True) ),
#	            
#	            # 512x3x3 - (512,1,3,1)
#	            ('conv5', nn.Conv2d(512, 1, 3, 1, bias=True) ),
#	            ('sig1', nn.Sigmoid() )
#            ])
	            nn.Conv2d(self.nc, 64, 3, 2, bias=True),
	            nn.BatchNorm2d(64),
	            nn.LeakyReLU(0.01,inplace=True),

                # 64x63x63 - (64,64,3,2)
                nn.MaxPool2d(3, 2),
	            
	            # 64x31x31 - (64,128,3,2)
	            nn.Conv2d(64, 128, 3, 2, bias=True),
	            nn.BatchNorm2d(128),
	            nn.LeakyReLU(0.01, inplace=True),

                # 128x15x15 - (128,128,3,2)
                nn.MaxPool2d(3, 2),
	            
	            # 128x7x7 - (128,256,3,1)
	            nn.Conv2d(128, 256, 3, 1, bias=True),
	            nn.BatchNorm2d(256),
	            nn.LeakyReLU(0.01,inplace=True),
	            
	            # 256x5x5 - (256,512,3,1)
	            nn.Conv2d(256, 512, 3, 1, bias=True),
	            nn.BatchNorm2d(512),
	            nn.LeakyReLU(0.01,inplace=True),
	            
	            # 512x3x3 - (512,1,3,1)
	            nn.Conv2d(512, 1, 3, 1, bias=True),
	            nn.Sigmoid()

        )
        
        # initialize weights
        self.netG = ModelBuilder().cuda().train()
        self.max_pool=nn.MaxPool1d(5*25*25,return_indices=True)
        self._initialize_weight() 
        self.lr_wh=nn.Parameter(0.99*torch.ones(2))

#        self.block_a = Block_A()
    def forward(self, data):
#        return self.discriminator(inputs)
    ################
#        Tensor = torch.cuda.FloatTensor
#        imgs = data['S'].cuda().float()
        bboxs = data['bb2'].cuda().float()
        bboxs_prev = data['bb1'].cuda().float()        
#        gt_patches=data['gt_patches'].cuda()
        zf=self.netG.backbone(data['template'].cuda())
        xf=self.netG.backbone(data['search'].cuda())
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()
        if cfg.ADJUST.ADJUST:
            xf = self.netG.neck(xf)
            zf = self.netG.neck(zf)
        clss, locs = self.netG.rpn_head(zf,xf)
        cls = self.netG.log_softmax(clss)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(locs, label_loc, label_loc_weight)
        batch = clss.shape[0]
        #################
#        pred_bboxs=fn(imgs,bboxs,clss,locs).reshape(batch,4)
        
#        val,idx=self.max_pool(cls[:,:,:,:,0])

        with torch.no_grad():
            anchors,window = my_bridge() #score_size

        def convert_bbox(locs, anchor):
            batch,_,_,_=locs.shape
            locs2=locs.permute(1,2,3,0).contiguous().view(batch,4,-1)
            delta=locs2.clone()
            delta[:,2,:] = torch.exp(locs2[:,2,:]) * anchor[:,2] 
            delta[:,3,:] = torch.exp(locs2[:,3,:]) * anchor[:,3] 
            delta[:,0,:] = locs2[:,0,:] * anchor[:,2] + anchor[:,0]
            delta[:,1,:] = locs2[:,1,:] * anchor[:,3] + anchor[:,1]
            return delta, batch
        
        def convert_score(clss):
            batch,_,_,_=clss.shape
            score = clss.permute(1, 2, 3, 0).contiguous().view(batch,2,-1).permute(0,2,1)
            score2 = F.softmax(score, dim=2)
            return score2[:,:,1]

        score = convert_score(clss)

        pred_bbox,batch = convert_bbox(locs, anchors)

        def change(r):
            return torch.max(r, 1. / r)
        def sz(w, h):
            pad = (w + h) * 0.5
            return torch.sqrt((w + pad) * (h + pad))

        size = [bboxs_prev[:,2], bboxs_prev[:,3]]

        w_z = size[0] + cfg.TRACK.CONTEXT_AMOUNT * (size[0]+size[1])
        h_z = size[1] + cfg.TRACK.CONTEXT_AMOUNT * (size[0]+size[1])
        s_z = torch.round(torch.sqrt(w_z * h_z))
        scale_z = torch.tensor(cfg.TRACK.EXEMPLAR_SIZE).cuda().float() / s_z

        # scale penalty
        s_c = change(sz(pred_bbox[:,2,:], pred_bbox[:,3,:]) /
                     (sz(size[0]*scale_z, size[1]*scale_z)).repeat(pred_bbox.shape[2],1).permute(1,0))
        # aspect ratio penalty
        r_c = change((size[0]/size[1]).repeat(pred_bbox.shape[2],1).permute(1,0) /
                     (pred_bbox[:,2,:]/pred_bbox[:,3,:]))
        penalty = torch.exp(-(r_c * s_c - 1) * torch.tensor(cfg.TRACK.PENALTY_K).cuda().float())        
        pscore = torch.mul(penalty,score)
#
#        # window penalty
        pscore = pscore * (1.0 - torch.tensor(cfg.TRACK.WINDOW_INFLUENCE).cuda().float()) + \
            window * torch.tensor(cfg.TRACK.WINDOW_INFLUENCE).cuda().float()
#        best_idx = torch.argmax(pscore)
        bboxs_new = pred_bbox[:,0:4,:]/ scale_z.repeat(pred_bbox.shape[2]*4,1).reshape(-1,4,pred_bbox.shape[2])
#
##        lr = torch.mul(pscore,torch.tensor(cfg.TRACK.LR).cuda())
        gt_score=torch.clone(score).fill_(0.0)
##        with torch.no_grad():
        val2,idx2=self.max_pool(pscore.reshape(batch,1,bboxs_new.shape[2]))
        idx2=idx2.reshape(batch)
        for i in range(batch):
            gt_score[i,idx2[i]]=1.0
#            
#        lr=torch.mul(pscore,gt_score*torch.tensor(cfg.TRACK.LR).cuda().float()).sum().div(batch)                        
##        print(lr)
#        # smooth bbox
        width = size[0].repeat(bboxs_new.shape[2],1).permute(1,0) * (1 - lr) + bboxs_new[:,2,:] * lr
        height = size[1].repeat(bboxs_new.shape[2],1).permute(1,0) * (1 - lr) + bboxs_new[:,3,:] * lr
#
        center_pos =[bboxs_prev[:,0]+(bboxs_prev[:,2]-1)/2,bboxs_prev[:,1]+(bboxs_prev[:,3]-1)/2]
#
        cx=center_pos[0].repeat(bboxs_new.shape[2],1).permute(1,0) + bboxs_new[:,0,:]      
        cy=center_pos[1].repeat(bboxs_new.shape[2],1).permute(1,0) + bboxs_new[:,1,:]
#
#        # clip boundary
##        cx, cy, width, height = _bbox_clip(cx, cy, width,
##                                                height, img.shape[:2])
##
        bboxs_updt =torch.stack( [cx - width / 2,
                cy - height / 2,
                width,
                height])

        IoU=iou(bboxs_updt.permute(1,0,2),bboxs.repeat(bboxs_updt.shape[2],1).reshape(-1,bboxs_updt.shape[2],4).permute(0,2,1))
#        gt_iou=torch.clone(IoU).fill_(0.0)

        val,idx=self.max_pool(IoU.reshape(batch,1,bboxs_updt.shape[2]))
        idx=idx.reshape(batch)

#        with torch.no_grad():
        for i in range(batch):
            gt_iou[i,idx[i]]=1.0
        iou_loss=1.0-torch.mul(IoU,gt_iou).sum().div(batch)
 #       template = data['template'].cuda()
 #       real = Variable(Tensor(template.size(0), 1, 1, 1).fill_(1.0), requires_grad=False)
 #       fake = Variable(Tensor(template.size(0), 1, 1, 1).fill_(0.0), requires_grad=False)
 #       lossfun = lossfn()
 #       real_loss = lossfun.adversarial_loss(self.discriminator(gt_patches), real,weight = None)
 #       fake_loss = lossfun.adversarial_loss(self.discriminator(gn_patches.detach()), fake, weight = None)
 #       dloss = (real_loss+fake_loss) / 2
 #       gloss = lossfun.adversarial_loss(self.discriminator(gn_patches), real,weight=None)
        outputs2 = {}
#        print(torch.argmax(IoU[0,:]),torch.argmax(pscore[0,:]))
        outputs2['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss 
#        outputs2['cls_loss'] = cls_loss
#        outputs2['loc_loss'] = loc_loss
#        outputs2['real_loss'] = real_loss
#        outputs2['fake_loss'] = fake_loss
#        outputs2['dloss']     = dloss
#        outputs2['gloss']     = gloss
        outputs2['iou_loss']  = iou_loss  
        return outputs2
#######################
    def _initialize_weight(self):
        """initialize network parameters"""
        initD = 'truncated'
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # kaiming initialization
#                if self.config.initD == 'kaiming':
#                    nn.init.kaiming_normal(m.weight.data, mode='fan_out')
#                    
#                # xavier initialization
#                elif self.config.initD == 'xavier':
#                   nn.init.xavier_normal(m.weight.data)
#                    m.bias.data.fill_(.1)
#
                if initD == 'truncated':
                    def truncated_norm_init(data, stddev=.01):
                        weight = np.random.normal(size=data.shape)
                        weight = np.clip(weight,
                                         a_min=-2*stddev, a_max=2*stddev)
                        weight = torch.from_numpy(weight).float()
                        return weight
                    m.weight.data = truncated_norm_init(m.weight.data)
                    #m.bias.data.fill_(.1)

                else:
                    raise NotImplementedError
            
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.momentum = 0.0003

#######################################################################################
class lossfn():
    def adversarial_loss(self, prediction, label, weight):
        #  weighted BCELoss 
        return F.binary_cross_entropy(prediction,
                                                  label,
                                                  weight)

    def mse_loss(self, prediction, label):
        return F.mse_loss(prediction, label)

    def hinge_loss(self, prediction, label):
        #  HingeEmbeddingLoss
        return F.hinge_embedding_loss(prediction,
                                     label,
                                     margin=1.0)

    def kldiv_loss(self, prediction, label):
        #  Kullback-Leibler divergence Loss.
        return F.kl_div(prediction,
                         label, 
                         reduction='batchmean')

    def weight_loss(self, prediction, label, weight):
        # weighted sigmoid cross entropy loss
        return F.binary_cross_entropy_with_logits(prediction.float(),label.float(),weight)

    def customize_loss(self, prediction, label, weight):
        score, y, weights = prediction, label, weight

        a = -(score * y)
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b) + torch.exp(a-b))
        loss = torch.mean(weights * loss)
        return loss
def build_data_loader():
#    logger.info("build train dataset")
    # train_dataset
    train_dataset = TrkDataset()
#    logger.info("build dataset done")

    train_sampler = None
#    if get_world_size() > 1:
#        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True,
                              sampler=train_sampler)
    return train_loader

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def iou(box_a,box_b, eps=1e-5):
    xi = torch.max(box_a[:,0], box_b[:,0])                                 # Intersection
    yi = torch.max(box_a[:,1], box_b[:,1])
    wi = torch.clamp(torch.min(box_a[:,2], box_b[:,2]) - xi, min=0)
    hi = torch.clamp(torch.min(box_a[:,3], box_b[:,3]) - yi, min=0)
    area_i = wi * hi                                       # Area Intersection
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])) # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1]))  # [A,B]
    area_u = area_a + area_b - area_i

    return area_i / torch.clamp(area_u, min=eps)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='siamrpn tracking')
    parser.add_argument('--cfg', type=str, default='../../experiments/siamrpn_r50_l234_dwxcorr_16gpu/config.yaml',
                    help='configuration of tracking')
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    train_loader = build_data_loader()
    netD=SiamNet_D().cuda().train()
    for idx, data in enumerate(train_loader):
        op=netD(data)
#        optimizer.zero_grad()
#        dloss = op['dloss']
#        gloss = op['gloss']
#        real_loss = op['real_loss'] 
#        fake_loss = op['fake_loss'] 
#        gloss.backward()
#        real_loss.backward()
#        fake_loss.backward()
        t=op['total_loss']
        l=op['iou_loss']
#        l.backward(retian_graph)
        (t+l).backward()
        if (idx%20) == 0:
            print("IOU_loss:",op['iou_loss'])
#        print(op['dloss']+op['gloss'])
#        print(op['real_loss']+op['fake_loss'])