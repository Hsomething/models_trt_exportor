# -*- coding: UTF-8 -*-
# @Time        :  2023-12-14 16:32
# @Author      :  Huangxiao
# @application :  models_trt_exportor
# @File        :  exportor.py
import os
import sys
import logging
import argparse
import tensorrt as trt
from dataloaders import *
from calibrator import Calibrator
from utils import get_file_createTime, get_fileSize

logger = logging.getLogger('testTracker')
handler = logging.StreamHandler()
logging_format = logging.Formatter(
    '[%(asctime)s,%(levelname)s,%(filename)s:%(lineno)s]: %(message)s')
handler.setFormatter(logging_format)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def check_list_version(version, min_version, split_char="."):
    version = [eval(item) for item in version.split(split_char)]
    min_version = [eval(item) for item in min_version.split(split_char)]
    max_len = max(len(version), len(min_version))
    version += [0] * (max_len - len(version))
    min_version += [0] * (max_len - len(min_version))
    ret = False
    for index in range(max_len - 1, -1, -1):
        if version[index] < min_version[index]:
            ret = False
        else:
            ret = True
    return ret


def main(in_file: str, workspace: int, test_size=(640, 640), fp16: bool = False, int8: bool = False,
         calib_dir: str = ''):
    if not os.path.exists(in_file):
        OSError(f"'{in_file}' not exists!")



    with trt.Logger(trt.Logger.INFO) as tlogger, trt.Builder(tlogger) as tbuilder, \
            tbuilder.create_builder_config() as config:

        network = tbuilder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, tlogger)
        if not parser.parse_from_file(in_file):
            raise RuntimeError(f'failed to load ONNX file: {in_file}')

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        logger.info(f"Input Shape of Model: {inputs[0].shape}")
        out_s = "Input Shape of Model: "
        for item in outputs:
            out_s += f"{item.shape} "
        logger.info(out_s)
        suffix=""
        if check_list_version(trt.__version__, '8.4'):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)
        else:
            config.max_workspace_size = workspace * 1 << 30
        if fp16:
            suffix="_fp16"
            logger.info(f"platform_has_fast_fp16: {tbuilder.platform_has_fast_fp16}")
            config.set_flag(trt.BuilderFlag.FP16)

        if int8:
            suffix = "_int8"
            logger.info(f"platform_has_fast_int8: {tbuilder.platform_has_fast_int8}")
            assert calib_dir, 'calib_dir is None or no image found'
            loader = YOLOV8DataLoader(calib_dir, 640, 100, 1)
            calib = Calibrator(loader, in_file.replace('.onnx', '_int8.cache'))
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calib

        if inputs[0].shape[0] != 1:
            profile = tbuilder.create_optimization_profile()
            for inp in inputs:
                # if inp.shape[0] == 1: continue
                profile.set_shape(inp.name, (1, 3, *test_size), (4, 3, *test_size), (4, 3, *test_size))
            config.add_optimization_profile(profile)

        out_engine = in_file.replace('.onnx', f'{suffix}.engine')
        with tbuilder.build_serialized_network(network, config) as engine, open(out_engine, 'wb') as t:
            t.write(engine)
        logger.info(f"TRT Model Export Success!  saved in '{out_engine}'")

def getArgs() -> argparse.Namespace:
    paraser = argparse.ArgumentParser()
    paraser.add_argument('-m', '--model', type=str, default='./yolo8s.pt',
                         help="the model to be export")
    paraser.add_argument('-w', '--workspace', type=int, default=1,
                         help="workspace of tensorrt, in sometime the bigger, the better of performance")
    paraser.add_argument('-s', '--imgsz', default="640",
                         help="test size of image")
    paraser.add_argument('-f16', '--fp16', default=False,action='store_true',
                         help="Whether use date type of float16 to speed inference")
    paraser.add_argument('-i8', '--int8', default=False, action='store_true',
                         help="Whether use date type of INT8 to speed inference")
    paraser.add_argument('-d', '--calib_dir', type=str, default='',
                         help="if int8 is choose, need some image to get scale and zero_point")

    logger.info(f"Running CMD: {' '.join(sys.argv)}")

    return paraser.parse_args()

if __name__ == "__main__":
    args = getArgs()
    logger.info(f"Args Value: {args}")
    assert os.path.exists(args.model), f"input model '{args.model}' is not exists!"
    logger.info(f"\nInput Model: {args.model} \n\size: {get_fileSize(args.model)} M \ncreate Time: {get_file_createTime(args.model)}")
    if args.fp16: logger.info("Use FP16 Mod")
    if args.int8:
        assert args.calib_dir and os.path.listdir(args.calib_dir) and os.path.exists(args.calib_dir), "calib_dir is Error!"
        logger.info("Use Int8 Model")
    test_size = [int(item) for item in args.imgsz.split(',')]
    if len(test_size) == 1:
        test_size = test_size*2
    main(args.model, args.workspace, test_size, fp16=args.fp16, int8=args.int8, calib_dir=args.calib_dir)