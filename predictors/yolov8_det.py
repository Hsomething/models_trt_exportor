# -*- coding: UTF-8 -*-
# @Time        :  2023-12-14 16:38
# @Author      :  Huangxiao
# @application :  models_trt_exportor
# @File        :  yolov8_infer.py
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

class YOLO8Predictor(object):

    def __init__(self, engien_path, test_size, cls_names, conf=0.2, nms_thres=0.5):
        self.cfx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()
        self.cls_names = cls_names
        self.conf = conf
        self.nms_thres = nms_thres
        self.color_palette = np.random.uniform(0, 255, size=(len(self.cls_names), 3))
        if isinstance(test_size, int):
            self.test_size = (test_size, test_size)
        else:
            self.test_size = test_size

        with open(engien_path, 'rb') as f, trt.Logger(trt.Logger.INFO) as tlogger, trt.Runtime(tlogger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # 分配主机和设备buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # 将设备buffer绑定到设备.
            self.bindings.append(int(cuda_mem))
            # 绑定到输入输出
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

    def pre_process(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
        # Get the height and width of the input image
        # img_height, img_width = img.shape[:2]
        origin_size = img.shape[:2]

        # Convert the image color space from BGR to RGB
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[:, :, ::-1]
        # Resize the image to match the input shape
        img = cv2.resize(img, self.test_size)

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data, origin_size

    def nms(self, boxes, scores, nms_thres):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thres)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, img, output, origin_size):
        outputs = np.transpose(np.squeeze(output))
        boxes = outputs[:, :4]
        scores = np.amax(outputs[:, 4:], axis=1)
        class_ids = np.argmax(outputs[:, 4:], axis=1)
        # Calculate the scaling factors for the bounding box coordinates
        x_factor = origin_size[1] / self.test_size[1]
        y_factor = origin_size[0] / self.test_size[0]

        valid_score_mask = scores > self.conf

        valid_scores = scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = class_ids[valid_score_mask]

        # x, y, w, h = valid_boxes[:, 0], valid_boxes[:, 1], valid_boxes[:, 2], valid_boxes[:, 3]
        boxes = np.zeros_like(valid_boxes)
        boxes[:, 0] = (valid_boxes[:, 0] - valid_boxes[:, 2] / 2) * x_factor
        boxes[:, 1] = (valid_boxes[:, 1] - valid_boxes[:, 3] / 2) * y_factor
        boxes[:, 2] = valid_boxes[:, 2] * x_factor + boxes[:, 0]
        boxes[:, 3] = valid_boxes[:, 3] * y_factor + boxes[:, 1]
        boxes.astype("int")

        indices = self.nms(boxes, valid_scores, self.nms_thres)
        if indices:
            dets = np.concatenate(
                [boxes[indices], valid_scores[indices, None], valid_cls_inds[indices, None]], 1
            )
        else:
            dets = []
        return dets

    def __call__(self, img):
        img, origin_size = self.pre_process(img)
        # print(img.shape)
        np.copyto(self.host_inputs[0], img.ravel())
        # 将输入数据转到GPU.
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        # 推理.
        self.cfx.push()
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        self.cfx.pop()
        # 将推理结果传到CPU.
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        # 同步 stream
        self.stream.synchronize()
        # 拿到推理结果 batch_size = 1
        output = self.host_outputs[0]
        start = time.time()
        ret = self.postprocess(img, output.reshape((4 + len(self.cls_names), -1)), origin_size)
        return ret

    def __del__(self):
        self.cfx.pop()

    def draw_bbox(self, img, box, score, class_id):
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

        # Create the label text with class name and score
        label = f'{self.cls_names[class_id]}: {score:.2f}'

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = int(box[0])
        label_y = int(box[1] - 10) if box[1] - 10 > label_height else int(box[1] + 10)

        # Draw a filled rectangle as the background for the label text
        print((label_x, label_y - label_height), (label_x + label_width, label_y + label_height))
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


if __name__ == "__main__":
    import yaml

    with open("/media/popsmart/big_disk/hx/ultralytics-main/ultralytics/cfg/datasets/coco128.yaml", errors='ignore',
              encoding='utf-8') as f:
        s = f.read()  # string
        data = yaml.safe_load(s)
    cls_names = data["names"]
    # model = YOLO8Inference("/media/popsmart/big_disk/hx/ultralytics-main/runs/detect/train/weights/best_int8.engine",
    model = YOLO8Predictor(
        "/media/popsmart/big_disk/hx/ultralytics-main/runs/detect/train/weights/best_optimization_level_5_int8_min-max_images64.engine",
        test_size=640, cls_names=cls_names
        )
    for i in range(10):
        ret = model("/media/popsmart/big_disk/hx/ultralytics-main/bus.jpg")

    model("/media/popsmart/big_disk/hx/ultralytics-main/bus.jpg")
    img = cv2.imread("/media/popsmart/big_disk/hx/ultralytics-main/bus.jpg")
    for x1, y1, x2, y2, score, cls_idx in ret:
        model.draw_bbox(img, [x1, y1, x2, y2], score, int(cls_idx))
    cv2.imwrite('bus_result_int8.jpg', img)
    # model.cfx.pop()