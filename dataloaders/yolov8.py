# -*- coding: UTF-8 -*-
# @Time        :  2023-12-14 16:33
# @Author      :  Huangxiao
# @application :  models_trt_exportor
# @File        :  yolov8.py
import os
import glob
import numpy as np
import cv2

class YOLOV8DataLoader:
    def __init__(self, img_dir, test_size, batch, batch_size, ):
        self.index = 0
        self.length = batch
        self.batch_size = batch_size
        if isinstance(test_size, int):
            self.test_size = (test_size, test_size)
        else:
            self.test_size = test_size
        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        self.img_list = glob.glob(os.path.join(img_dir, "*.jpg"))
        assert len(self.img_list) > self.batch_size * self.length, '{} must contains more than '.format(
            img_dir) + str(self.batch_size * self.length) + ' images to calib'
        print('found all {} images to calib.'.format(len(self.img_list)))
        self.calibration_data = np.zeros((self.batch_size, 3, *self.test_size), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = self.pre_process(img)
                self.calibration_data[i] = img

            self.index += 1

            # example only
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def pre_process(self, img):
        # origin_size = img.shape[:2]
        img = img[:, :, ::-1]
        img = cv2.resize(img, self.test_size)
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        # image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data

    def __len__(self):
        return self.length
