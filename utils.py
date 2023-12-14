# -*- coding: UTF-8 -*-
# @Time        :  2023-12-14 17:11
# @Author      :  Huangxiao
# @application :  models_trt_exportor
# @File        :  utils.py
import os
import time
def get_file_createTime(in_file):
    create_time = os.path.getctime(in_file)
    return time.strftime("%Y-%m-%d %H:%M%S", time.localtime(create_time))
def get_fileSize(int_file):
    fsize = os.path.getsize(int_file)
    fsize = fsize / float(1024 * 1024)
    return round(fsize, 2)