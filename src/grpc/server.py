'''
@Author: captainfffsama
@Date: 2023-02-02 15:59:46
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-02-02 16:43:55
@FilePath: /LoFTR/src/grpc/server.py
@Description:
'''
import numpy as np

from src.loftr import default_cfg

from . import loftr_pb2
from .loftr_pb2_grpc import LoftrServicer

from .loftr_worker import LoFTRWorker
from .utils import decode_img_from_proto, np2tensor_proto, img2pb_img


class LoFTRServer(LoftrServicer):
    def __init__(self,
                 ckpt_path,
                 device="cuda:0",
                 thr=0.5,
                 ransc_method="USAC_MAGSAC",
                 ransc_thr=3,
                 ransc_max_iter=2000,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.worker = LoFTRWorker(
            default_cfg,
            ckpt_path,
            device,
            thr,
            ransc_method,
            ransc_thr,
            ransc_max_iter,
        )

    def getEssentialMatrix(self, request, context):
        imgA = decode_img_from_proto(request.imageA)
        imgB = decode_img_from_proto(request.imageB)
        if imgA is None or imgB is None:
            return loftr_pb2.GetEssentialMatrixReply(matrix=np2tensor_proto(
                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float)),
                                                    status=-14)
        H, flag = self.worker(imgA, imgB)
        status = 0 if flag else -14
        return loftr_pb2.GetEssentialMatrixReply(matrix=np2tensor_proto(H),
                                                status=status)
