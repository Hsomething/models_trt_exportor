# models_trt_exportor

this repository is the script of Convert, which convert onnx Model to TRT. Support TRT Model Mode:

- FP16
- INT8

 Support Model List

​	:ballot_box_with_check: YOLOV8

​	:ballot_box_with_check: YOLOX

TODO List:

- [ ] check inference speed of int8

## How To Use ?

1. you need a Environment of Python, and `Pyhon >= 3.8`. Can use Conda to create env

    ```shell
    conda create -n trt_exporter python=3.8
    conda activate trt_exporter
    git clone git@github.com:Hsomething/models_trt_exportor.git
    cd models_trt_exportor && pip install -r requirements.txt
    ```

    

2. Install TensorRT to Python

3. Run And Export

    ```shell
    # python exporter.py -m /path/your/onnx/model.onnx -w 2 -s 640 
    python exporter.py -m ./yolo8s.onnx -w 2 -s 640 
    # fp16
    python exporter.py -m ./yolo8s.onnx -w 2 -s 640 -fp16
    # int8
    python exporter.py -m ./yolo8s.onnx -w 2 -s 640 -int8 -d /dir/of/train/set/images
    
    # In same level of model you will get yolo8s.engine/yolo8s_fp16.engine/yolo8sint8.engine
    ```

    

## Infer Times

| Model structure | Backend           | test times | Language | Pipeline Avg Time(ms) | infer Avg Time(ms) |
| --------------- | ----------------- | ---------- | -------- | --------------------- | ------------------ |
| YOLO8-s         | PyTorch           | 100        | Python   | 21.5                  | 6.3                |
| YOLO8-s         | PyTorch(Half)     | 100        | Python   | 19.95                 | 4.8                |
| YOLO8-s         | OnnxRuntime       | 100        | Python   | 62.7                  | 8.2                |
| YOLO8-s         | OnnxRuntime(Half) | 100        | Python   | 65.48                 | 8.1                |
| YOLO8-s         | TensorRT          | 100        | Python   | 27.06                 | 6.2                |
| YOLO8-s         | TensorRT(Half)    | 100        | Python   | 24.31                 | 3.72               |

