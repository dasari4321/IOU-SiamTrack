META_ARC: "siamrpn_alex_dwxcorr"

BACKBONE:
    TYPE: "alexnet"
    KWARGS:
        width_mult: 1.0   
    PRETRAINED: 'pretrained_models/alexnet-bn.pth'
    TRAIN_LAYERS: ['layer4', 'layer5']
    TRAIN_EPOCH: 10
    LAYERS_LR: 1.0

ADJUST:
    ADJUST: False

RPN:
    TYPE: 'DepthwiseRPN'
    KWARGS:
        anchor_num: 5
        in_channels: 256
        out_channels: 256

MASK:
    MASK: False

ANCHOR:
    STRIDE: 8
    RATIOS: [0.33, 0.5, 1, 2, 3]
    SCALES: [8]
    ANCHOR_NUM: 5

TRACK:
    TYPE: 'SiamRPNTracker'
    PENALTY_K: 0.16
    WINDOW_INFLUENCE: 0.40
    LR: 0.30
    EXEMPLAR_SIZE: 127
    INSTANCE_SIZE: 287
    BASE_SIZE: 0
    CONTEXT_AMOUNT: 0.5

TRAIN:
    EPOCH: 50
    START_EPOCH: 0
    BATCH_SIZE: 8
    BASE_SIZE: 0
    OUTPUT_SIZE: 17
    BASE_LR: 0.005
    CLS_WEIGHT: 1.
    LOC_WEIGHT: 1.2
    RESUME: ''

    LR:
        TYPE: 'log'
        KWARGS:
            start_lr: 0.01
            end_lr: 0.0005
    LR_WARMUP:
        TYPE: 'step'
        EPOCH: 5
        KWARGS:
            start_lr: 0.005
            end_lr: 0.01
            step: 1

DATASET:
    NAMES:
    - 'VID'
   # - 'YOUTUBEBB'
   # - 'COCO'
   # - 'DET'

    TEMPLATE:
        SHIFT: 4
        SCALE: 0.05
        BLUR: 0.0
        FLIP: 0.0
        COLOR: 1.0

    SEARCH:
        SHIFT: 64
        SCALE: 0.18
        BLUR: 0.2
        FLIP: 0.0
        COLOR: 1.0

    NEG: 0.05
    GRAY: 0.0
