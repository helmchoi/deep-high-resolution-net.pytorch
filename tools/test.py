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
from core.function import validate
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

def get_engine(onnx_file_path, engine_file_path=""):
    def build_engine():
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30 # 1GB
            builder.max_batch_size = 1
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()
        

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

    if (cfg.MODEL.NAME == 'faster_vit_4_21k_224'):
        # model = models.create_model(cfg.MODEL.NAME, 
        #                             pretrained=True,
        #                             heatmap_sizes=(90,90),
        #                             model_path="models/pytorch/imagenet/fastervit_4_21k_224_w14.pth.tar")
        model = create_model(cfg.MODEL.NAME,
                            pretrained=False,
                            heatmap_sizes=(90,90),
                            kernel_size=9,
                            scriptable=True)
    else:
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

    if (cfg.MODEL.NAME == 'faster_vit_4_21k_224'):
        input = torch.randn(1,3,224,224).cuda()
        test_input = torch.ones(1,3,224,224).cuda()
    else:
        input = torch.randn(1,3,256,256).cuda()
        test_input = torch.ones(1,3,256,256).cuda()

    # torch.onnx.export(model,               # 실행될 모델
    #                     input,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
    #                     "dummy.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
    #                     export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
    #                     opset_version=11,          # 모델을 변환할 때 사용할 ONNX 버전
    #                     do_constant_folding=True,  # 최적하시 상수폴딩을 사용할지의 여부
    #                     input_names = ['input'],   # 모델의 입력값을 가리키는 이름
    #                     output_names = ['output'], # 모델의 출력값을 가리키는 이름
    #                     dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
    #                                     'output' : {0 : 'batch_size'}})
                                
    # get_engine("dummy.onnx", engine_file_path="dummy.engine")

    trace_v = torch.jit.trace(model, input)
    trace_v.save("output/unreal/faster_vit_4_21k_224/dummy.pt")
    # with torch.no_grad():
    #     torch.jit.save(trace_v, "/home/inrol/Downloads/trace_Unreal.pt")
    test_output = model(test_input)
    trace_output = trace_v(test_input)
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


if __name__ == '__main__':
    main()
