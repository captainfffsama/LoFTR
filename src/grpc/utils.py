# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-02-02 16:00:45
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-02-02 16:03:15
@FilePath: /LoFTR/src/grpc/utils.py
@Description:
'''
import os
import base64

import numpy as np
import cv2

from . import loftr_pb2


def get_img(img_info):
    if os.path.isfile(img_info):
        if not os.path.exists(img_info):
            return None
        else:
            return cv2.imread(img_info)  #ignore
    else:
        img_str = base64.b64decode(img_info)
        img_np = np.fromstring(img_str, np.uint8)
        return cv2.imdecode(img_np, cv2.IMREAD_COLOR)


def decode_img_from_proto(proto_image):
    if proto_image.image:
        return get_img(proto_image.image)
    else:
        return get_img(proto_image.path)


def np2tensor_proto(np_ndarray: np.ndarray):
    shape = list(np_ndarray.shape)
    data = np_ndarray.flatten().tolist()
    tensor_pb = loftr_pb2.Tensor()
    tensor_pb.shape.extend(shape)
    tensor_pb.data.extend(data)
    return tensor_pb


def img2pb_img(img):
    base64_str = cv2.imencode('.jpg', img)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    return loftr_pb2.Image(image=base64_str)


def tensor_proto2np(tensor_pb):
    np_matrix = np.array(tensor_pb.data,
                         dtype=np.float).reshape(tensor_pb.shape)
    return np_matrix