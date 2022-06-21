import __init_paths

import torch
#import numpy 
import torch.nn as nn
from torch.autograd import Function
from autograd import grad

import autograd.numpy as np
from autograd import elementwise_grad as egrad
from torch.nn import functional as F

from pysot.utils.anchor import Anchors
from pysot.core.config import cfg
import cv2

Tensor = torch.cuda.FloatTensor
def get_subwindow(im, pos, model_sz, original_sz, avg_chans):
    """
    args:
        im: bgr based image
        pos: center position
        model_sz: exemplar size
        s_z: original size
        avg_chans: channel average
    """
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    # context_xmin = round(pos[0] - c) # py2 and py3 round
    context_xmin = np.floor(pos[0] - c + 0.5)
    context_xmax = context_xmin + sz - 1
    # context_ymin = round(pos[1] - c)
    context_ymin = np.floor(pos[1] - c + 0.5)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
        te_im = np.zeros(size, np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                         int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch = im[int(context_ymin):int(context_ymax + 1),
                      int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch, (model_sz, model_sz))
    im_patch = im_patch.transpose(2, 0, 1)
    im_patch = im_patch[np.newaxis, :, :, :]
    im_patch = im_patch.astype(np.float32)
    im_patch = torch.from_numpy(im_patch)
    return im_patch

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
    pred_bboxs=[]
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
        center_pos = np.array([bbox[0]+(bbox[2]-1)/2,bbox[1]+(bbox[3]-1)/2])
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
        gn_patche = get_subwindow(img2, center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, channel_average)
        if cfg.CUDA:
            gn_patche = gn_patche.cuda()
        gn_patches.append(Tensor(gn_patche))
        pred_bboxs.append(Tensor(bbox))
    gn_patches=torch.cat(gn_patches)
    return gn_patches

class _Block_A(Function):
    @staticmethod
    def forward(ctx,imgs,bboxs,clss,locs):
        gn_patches=fn(imgs,bboxs,clss,locs)
        return gn_patches
    @staticmethod
    def backward(ctx,grad_output):
        
        return None, None, None, None

my_block=_Block_A.apply

class Block_A(nn.Module):
    def __init__(self):
        super(Block_A, self).__init__()

    def forward(self,imgs,bboxs,clss,locs):
        return my_block(imgs,bboxs,clss,locs)

if __name__=='__main__':
    blk=Block_A()
    img = torch.ones(8,360,240,3).cuda()
    bbox= torch.from_numpy(np.array([[3,3,49,49]]*8))
    clss=torch.rand(8,10,25,25).cuda()
    locs=torch.rand(8,20,25,25).cuda()
    draft=torch.ones(8,3,127,127,dtype=torch.float32,requires_grad=True)
    draft=blk(img,bbox,clss,locs)
    