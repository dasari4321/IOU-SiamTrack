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
from pysot.tracker.base_tracker import SiameseTracker
from pysot.utils.anchor import Anchors
from pysot.datasets.dataset import TrkDataset
from torch.utils.data import DataLoader
from pysot.models.model_builder import ModelBuilder
########################################################################
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

class SiamNet_D(nn.Module):
    def __init__(self):
        super(SiamNet_D, self).__init__()
#        self.config = Config()
        self.nc = 3
        self.discriminator = nn.Sequential(
            
            OrderedDict([
	            # 3x127x127 - (3,64,3,2)
	            ('conv1', nn.Conv2d(self.nc, 64, 3, 2, bias=True) ),
	            ('bn1',	nn.BatchNorm2d(64) ),
	            ('lrelu1', nn.LeakyReLU(0.01,inplace=True) ),

                # 64x63x63 - (64,64,3,2)
                ('pool1', nn.MaxPool2d(3, 2)),
	            
	            # 64x31x31 - (64,128,3,2)
	            ('conv2', nn.Conv2d(64, 128, 3, 2, bias=True) ),
	            ('bn2',	nn.BatchNorm2d(128) ),
	            ('lrelu2', nn.LeakyReLU(0.01, inplace=True) ),

                # 128x15x15 - (128,128,3,2)
                ('pool2', nn.MaxPool2d(3, 2)),
	            
	            # 128x7x7 - (128,256,3,1)
	            ('conv3', nn.Conv2d(128, 256, 3, 1, bias=True) ),
	            ('bn3',	nn.BatchNorm2d(256) ),
	            ('lrelu3', nn.LeakyReLU(0.01,inplace=True) ),
	            
	            # 256x5x5 - (256,512,3,1)
	            ('conv4', nn.Conv2d(256, 512, 3, 1, bias=True) ),
	            ('bn4',	nn.BatchNorm2d(512) ),
	            ('lrelu4', nn.LeakyReLU(0.01,inplace=True) ),
	            
	            # 512x3x3 - (512,1,3,1)
	            ('conv5', nn.Conv2d(512, 1, 3, 1, bias=True) ),
	            ('sig1', nn.Sigmoid() )
            ])
        )
        
        # initialize weights
        self._initialize_weight() 
        self.netG = ModelBuilder()
        
    def forward(self, data):
#        return self.discriminator(inputs)
    ################
        Tensor = torch.cuda.FloatTensor
        imgs = data['S']
        bboxs =data['bb2']
        gt_patches=[]
        gn_patches=[]
        trk = SiameseTracker()
        for img, bbox in zip(imgs, bboxs):
            
            center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
            size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
            w_z = size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
            h_z = size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
            s_z = round(np.sqrt(w_z * h_z))
            scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
            # calculate channle average
            img2=img.squeeze().cpu().numpy()
            channel_average = np.mean(img2, axis=(0, 1))
            # get crop
    
            ####$$$$$$$$$$$$$$$$$
            gt_patche = trk.get_subwindow(img2, center_pos,
                                        cfg.TRACK.EXEMPLAR_SIZE,
                                        s_z, channel_average)
            gt_patches.append(Tensor(gt_patche))
        gt_patches=torch.cat(gt_patches)
        ####$$$$$$$$$$$$$$$$$
#        score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
#            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(25) #score_size
        window = np.outer(hanning, hanning)
        window = np.tile(window.flatten(), anchor_num)
        anchors = generate_anchor(score_size=25) #score_size
        outputs, clss, locs = self.netG(data)
        zf=self.netG.backbone(data['template'].cuda())
        xf=self.netG.backbone(data['search'].cuda())

        if cfg.ADJUST.ADJUST:
            xf = self.netG.neck(xf)
            zf = self.netG.neck(zf)
        clss, locs = self.netG.rpn_head(zf,xf)
        for cls,loc in zip(clss,locs):
            score = _convert_score(cls.unsqueeze(0))
            pred_bbox = _convert_bbox(loc.unsqueeze(0), anchors)
   
            def change(r):
                return np.maximum(r, 1. / r)
    
            def sz(w, h):
                pad = (w + h) * 0.5
                return np.sqrt((w + pad) * (h + pad))
    
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
        ####$$$$$$$$$$$$$
        template = data['template'].cuda()
        real = Variable(Tensor(template.size(0), 1, 1, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(template.size(0), 1, 1, 1).fill_(0.0), requires_grad=False)
        lossfun = lossfn()
        real_loss = lossfun.adversarial_loss(self.discriminator(gt_patches), real,weight = None)
        fake_loss = lossfun.adversarial_loss(self.discriminator(gn_patches.detach()), fake, weight = None)
        dloss = (real_loss+fake_loss) / 2
        gloss = lossfun.adversarial_loss(self.discriminator(gn_patches), real,weight=None)
        outputs2 = {}
        outputs2['real_loss'] = real_loss
        outputs2['fake_loss'] = fake_loss
        outputs2['dloss']     = dloss
        outputs2['gloss']     = gloss
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
        print(op['real_loss']+op['fake_loss'])