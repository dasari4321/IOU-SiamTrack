import __init_paths

import autograd.numpy as np
from autograd import grad
#from autograd import elementwise_grad as egrad
from torch.nn import functional as F

from pysot.utils.anchor import Anchors
from pysot.core.config import cfg
from pysot.tracker.base_tracker import SiameseTracker
import torch 
import torch.nn as nn
from torch.autograd import Function

trk= SiameseTracker()
Tensor = torch.cuda.FloatTensor

#def fn(x):                 # Define a function
#    y = np.exp(-2.0 * x)
#    return y 

def generate_anchor(score_size=25):
    anchors = Anchors(cfg.ANCHOR.STRIDE,
                      cfg.ANCHOR.RATIOS,
                      cfg.ANCHOR.SCALES)
    anchor = anchors.anchors
    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
    total_stride = anchors.stride
    anchor_num = anchor.shape[0]
    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size // 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
        np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor

def _convert_bbox(delta, anchor):
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
    delta = delta.data.cpu().numpy()

    delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
    delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
    return delta

def _convert_score(score):
    score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
    score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
    return score

def _bbox_clip(cx, cy, width, height, boundary):
    cx = max(0, min(cx, boundary[1]))
    cy = max(0, min(cy, boundary[0]))
    width = max(10, min(width, boundary[1]))
    height = max(10, min(height, boundary[0]))
    return cx, cy, width, height

def fn(imgs,bboxs,clss,locs): 
    gn_patches=[]
    anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
    hanning = np.hanning(25) #score_size
    window = np.outer(hanning, hanning)
    window = np.tile(window.flatten(), anchor_num)
    anchors = generate_anchor(score_size=25) #score_size
    for img,bbox,cls,loc in zip(imgs,bboxs,clss,locs):
        score = _convert_score(cls.unsqueeze(0))
        pred_bbox = _convert_bbox(loc.unsqueeze(0), anchors)
   
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))
        center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                bbox[1]+(bbox[3]-1)/2])
        size = np.array([bbox[2], bbox[3]])    
        w_z = size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
        h_z = size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
        s_z = round(np.sqrt(w_z * h_z))

        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(size[0]*scale_z, size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((size[0]/size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + center_pos[0]
        cy = bbox[1] + center_pos[1]

        # smooth bbox
        width = size[0] * (1 - lr) + bbox[2] * lr
        height = size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = _bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        center_pos = np.array([cx, cy])
        size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
#        best_score = score[best_idx]
        center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
        h_z = size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        img2=img.squeeze().cpu().numpy()
        channel_average = np.mean(img2, axis=(0, 1))    
        # get crop
        ####$$$$$$$$$$$$$
        gn_patche = trk.get_subwindow(img2, center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, channel_average)
        gn_patches.append(Tensor(gn_patche))
    gn_patches=torch.cat(gn_patches)
    return gn_patches

class _Block_A(Function):
    def forward(ctx,imgs,bboxs,clss,locs):
        gn_patches=fn(imgs,bboxs,clss,locs)
        return gn_patches

    def backward(ctx,grad_output):
        grad_fn1=grad(fn,2)
        grad_fn2=grad(fn,3)
        grad_cls=grad_fn1(grad_output)
        grad_loc=grad_fn2(grad_output)
        return None,None,grad_cls,grad_loc

my_block=_Block_A.apply

class Block_A(nn.Module):
    def __init__(self, imgs,bboxs):
        super(Block_A, self).__init__()
        self.imgs = imgs 
        self.bboxs = bboxs

    def forward(self,clss,locs):
        return my_block(self.imgs, self.bboxs,clss,locs)
#
#out=fn(1.0)
#grad_out=out
#grad_fn = grad(fn)       # Obtain its gradient function
#grad_in=grad_fn(grad_out)               # Evaluate the gradient at x = 1.0


#######################################
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#import torch
#from torch import nn
#from torch.autograd import Function
#from torch.autograd.function import once_differentiable
#from torch.nn.modules.utils import _pair
#
#from model import _C
#
#
#class _ROIPool(Function):
#    @staticmethod
#    def forward(ctx, input, roi, output_size, spatial_scale):
#        ctx.output_size = _pair(output_size)
#        ctx.spatial_scale = spatial_scale
#        ctx.input_shape = input.size()
#        output, argmax = _C.roi_pool_forward(
#            input, roi, spatial_scale, output_size[0], output_size[1]
#        )
#        ctx.save_for_backward(input, roi, argmax)
#        return output
#
#    @staticmethod
#    @once_differentiable
#    def backward(ctx, grad_output):
#        input, rois, argmax = ctx.saved_tensors
#        output_size = ctx.output_size
#        spatial_scale = ctx.spatial_scale
#        bs, ch, h, w = ctx.input_shape
#        grad_input = _C.roi_pool_backward(
#            grad_output,
#            input,
#            rois,
#            argmax,
#            spatial_scale,
#            output_size[0],
#            output_size[1],
#            bs,
#            ch,
#            h,
#            w,
#        )
#        return grad_input, None, None, None
#
#
#roi_pool = _ROIPool.apply
#
#
#class ROIPool(nn.Module):
#    def __init__(self, output_size, spatial_scale):
#        super(ROIPool, self).__init__()
#        self.output_size = output_size
#        self.spatial_scale = spatial_scale
#
#    def forward(self, input, rois):
#        return roi_pool(input, rois, self.output_size, self.spatial_scale)
#
#    def __repr__(self):
#        tmpstr = self.__class__.__name__ + "("
#        tmpstr += "output_size=" + str(self.output_size)
#        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
#        tmpstr += ")"
#        return tmpstr