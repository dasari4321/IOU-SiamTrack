#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:32:21 2020

@author: ee18d001
"""
#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#import __init_paths

import os
import argparse
import cv2
import sys
import vot
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.model_load import load_pretrain


class IGSTracker(object):
    def __init__(self,imagefile,image,bbox):
        cwd=os.path.dirname(os.path.abspath(__file__))
 #       print(cwd)
        
        config_file = os.path.join(cwd,'config.yaml')
        
        model_file = os.path.join(cwd,'model.pth')
        
        parser = argparse.ArgumentParser(description='IGS tracking')
        
        parser.add_argument('--config', default=config_file, type=str,
                help='config file')
        
        parser.add_argument('--model', default=model_file, type=str,
        
                            help='model file')
        args = parser.parse_args()
        
        cfg.merge_from_file(args.config)

        # create model
        model = ModelBuilder()
        
        # load model
        model = load_pretrain(model, args.model).cuda().eval()
        
        # build tracker
        self.tracker = build_tracker(model)

        #initialise
        self.tracker.init(image, bbox)

    def track(self,image):

        output=self.tracker.track(image)

        return output
        
handle = vot.VOT("rectangle")
selection = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)
image = cv2.imread(imagefile)

trkr = IGSTracker(imagefile,image, selection)
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)
    output = trkr.track(image)
    bb=output['bbox']
    region=vot.Rectangle(bb[0],bb[1],bb[2],bb[3])
    confidence=output['best_score']
    
    handle.report(region, confidence)

