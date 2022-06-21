# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck
import torch
from collections import OrderedDict
#from pysot.models.block_a1 import Block_A, fn

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        self.ATTN=False
        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)
        if self.ATTN:
            self.attn =  nn.Sequential(
                OrderedDict([
                    # 3x127x127 - (3,64,3,2)
                    ('conv1', nn.Conv2d(256, 64, 3, 1, 1, bias=True) ),
                    ('bn1', nn.BatchNorm2d(64) ),
                    ('lrelu1', nn.LeakyReLU(0.01,inplace=True) ),
                    ('conv2', nn.Conv2d(64, 16, 3, 1, 1, bias=True) ),
                    ('bn2', nn.BatchNorm2d(16) ),
                    ('lrelu1', nn.LeakyReLU(0.01,inplace=True) ),
                    ('conv5', nn.Conv2d(16, 1, 3, 1, 1, bias=True) ),
                    ('sig1', nn.Sigmoid() )
                ])
                )
        self.CONTEXT_AMOUNT=nn.Parameter(cfg.TRACK.CONTEXT_AMOUNT*torch.ones(1));
        self.PENALTY_K=nn.Parameter(cfg.TRACK.PENALTY_K*torch.ones(1));
        self.WINDOW_INFLUENCE=nn.Parameter(cfg.TRACK.WINDOW_INFLUENCE*torch.ones(1));
        self.LR=nn.Parameter(cfg.TRACK.LR*torch.ones(1));


    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if self.ATTN:
            zf = torch.mul(zf, self.attn(zf))
            xf = torch.mul(xf, self.attn(xf))
        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)
        
        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)
###############################################
#        with torch.no_grad():
#            IoU_loss = 
#        rpn_cls_score_reshape = self.reshape(cls, 2)
#        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
#        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)
#        base_feat = xf
#        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
#        im_info = 
#        cfg_key = 'TRAIN'
#        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)
#        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
#                                 im_info, cfg_key))
################################################
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss
        return outputs
