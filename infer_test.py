# -*- coding: UTF-8 -*-
# @Time        :  2023-12-14 16:43
# @Author      :  Huangxiao
# @application :  models_trt_exportor
# @File        :  infer_test.py
import time
import yaml
from ultralytics import YOLO
from infer_engine import YOLO8Inference
from infer_onnx import YOLOv8


def main():
    test_times = 100
    with open("/media/popsmart/big_disk/hx/ultralytics-main/ultralytics/cfg/datasets/coco128.yaml", errors='ignore',
              encoding='utf-8') as f:
        s = f.read()  # string
        data = yaml.safe_load(s)
    cls_names = data["names"]
    img_path = "/media/popsmart/big_disk/hx/ultralytics-main/bus.jpg"
    tr_model = YOLO("/media/popsmart/big_disk/hx/ultralytics-main/runs/detect/train/weights/best.pt")
    engine_model = YOLO8Inference("/media/popsmart/big_disk/hx/ultralytics-main/runs/detect/train/weights/best.engine",
                                  640, cls_names)
    onnx_model = YOLOv8("/media/popsmart/big_disk/hx/ultralytics-main/runs/detect/train/weights/best.onnx",
                        img_path, 0.2, 0.5
                        )
    for i in range(10):
        tr_model.predict(img_path, save=False)

    start = time.time()
    for i in range(test_times):
        tr_model.predict(img_path, save=False)
    print("torch model Inference cost: ", time.time() - start)

    for i in range(10):
        engine_model(img_path)

    start = time.time()
    for i in range(test_times):
        engine_model(img_path)
    print("engine model Inference cost: ", time.time() - start)

    for i in range(10):
        onnx_model.main()

    start = time.time()
    for i in range(test_times):
        onnx_model.main()
    print("engine model Inference cost: ", time.time() - start)


if __name__ == "__main__":
    main()
