# -*- coding: UTF-8 -*-
# @Time        :  2023-12-14 16:43
# @Author      :  Huangxiao
# @application :  models_trt_exportor
# @File        :  yolox_det.py
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

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

class YOLOXPredictor(object):
    def __init__(self, engien_path, test_size, cls_names, conf=0.2, nms_thres=0.5):
        self.cfx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()
        self.cls_names = cls_names
        self.conf = conf
        self.nms_thres = nms_thres
        self.color_palette = np.random.uniform(0, 255, size=(len(self.cls_names), 3))
        self.infer_method = "execute_async"
        if check_list_version(trt.__version__, '8.4'):
            self.infer_method = "execute_async_v2"
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

    def pre_process(self, img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def postprocess(self, outputs, img_size, p6=False):
        grids = []
        expanded_strides = []
        strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs

    def boxes_translate(self, boxes, ratio):
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        return boxes_xyxy

    def multiclass_nms(self, boxes, scores, nms_thr, score_thr, class_agnostic=True):
        if class_agnostic:
            nms_method = self.multiclass_nms_class_agnostic
        else:
            nms_method = self.multiclass_nms_class_aware
        return nms_method(boxes, scores, nms_thr, score_thr)

    def multiclass_nms_class_aware(self,boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-aware version."""
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > score_thr
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self.nms(valid_boxes, valid_scores, nms_thr)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                    )
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)

    def multiclass_nms_class_agnostic(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-agnostic version."""
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self.nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
        return dets

    def nms(self, boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
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

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep
    def __call__(self, img):
        img, origin_ratio = self.pre_process(img, self.test_size)
        np.copyto(self.host_inputs[0], img.ravel())
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.cfx.push()
        eval(f"self.context.{self.infer_method}")(bindings=self.bindings, stream_handle=self.stream.handle)
        self.cfx.pop()
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        output = self.host_outputs[0]

        predictions = self.postprocess(output[0], self.test_size)

        boxes = predictions[:, :4]
        boxes_xyxy = self.boxes_translate(boxes, origin_ratio)
        scores = predictions[:, 4:5] * predictions[:, 5:]

        dets = self.multiclass_nms(boxes_xyxy, scores, nms_thr=self.nms_thres, score_thr=self.conf)