# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from timm.models import create_model

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate, validate_TC
from utils.utils import create_logger

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # if cfg.MODEL.NAME == 'faster_vit_4_21k_224' or cfg.MODEL.NAME == 'faster_vit_0_224':
    #     model = create_model(cfg.MODEL.NAME,
    #                         pretrained=False,
    #                         heatmap_sizes=(54,54),
    #                         kernel_size=3,
    #                         scriptable=True)
    # else:
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))
    
    # SAVE TORCH TRACE SCRIPT -------------------------------
    model.cuda()
    model.eval()

    # if cfg.MODEL.NAME == 'faster_vit_4_21k_224' or cfg.MODEL.NAME == 'faster_vit_0_224':
    #     input = torch.randn(1,3,224,224).cuda()
    #     test_input = torch.ones(1,3,224,224).cuda()
    # else:
    input = torch.randn(1,3,256,256).cuda()
    test_input = torch.ones(1,3,256,256).cuda()

    torch.onnx.export(model,               # 실행될 모델
                        input,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                        "output/unreal/pose_fastvit/w32_256x256_adam_lr1e-3_Unreal/240604_1007_fastvit_unreal_scale.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                        export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                        opset_version=11,          # 모델을 변환할 때 사용할 ONNX 버전
                        do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                        input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                        output_names = ['output'], # 모델의 출력값을 가리키는 이름
                        dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                        'output' : {0 : 'batch_size'}})

    input_hm = torch.randn(1,16,64,64).cuda()

    trace_v = torch.jit.trace(model, input)
    # trace_v = torch.jit.trace(model, (input, input_hm))
    trace_v.save("output/unreal/pose_fastvit/w32_256x256_adam_lr1e-3_Unreal/240604_1007_fastvit_unreal_scale.pt")
    # with torch.no_grad():
    #     torch.jit.save(trace_v, "/home/inrol/Downloads/trace_Unreal.pt")
    test_output = model(test_input)
    trace_output = trace_v(test_input)
    # test_output, _, _ = model(test_input, input_hm)
    # trace_output, _, _ = trace_v(test_input, input_hm)
    print("output shape: ", np.shape(test_output), np.shape(trace_output))
    print("model: ", test_output[0,5,35:40,30:35])
    print("trace: ", trace_output[0,5,35:40,30:35])
    # --------------------------------------------------------

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)
    # validate_TC(cfg, valid_loader, valid_dataset, model, criterion,
    #          final_output_dir, tb_log_dir)


if __name__ == '__main__':
    main()
