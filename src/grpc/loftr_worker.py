# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2023年 02月 02日 星期四 15:19:14 CST
@Description:
'''
from collections import namedtuple

import torch
import cv2
import numpy as np

from src.loftr import LoFTR, default_cfg

DebugInfo=namedtuple("DebugInfo",
                        ["kp0_fake_match","kp1_fake_match",
                        "kp0_true_match","kp1_true_match"])

class LoFTRWorker(object):

    def __init__(self,
                 config,
                 ckpt_path,
                 device="cuda:0",
                 thr=0.5,
                 ransc_method="USAC_MAGSAC",
                 ransc_thr=3,
                 ransc_max_iter=2000):
        self.model = LoFTR(config=config)
        self.model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        if device != 'cpu' and not torch.cuda.is_available():
            device = 'cpu'
            print("ERROR: cuda can not use, will use cpu")
        self.model = self.model.eval().to(device)
        self.thr=thr
        self.ransc_method = getattr(cv2,ransc_method)
        self.ransc_thr=ransc_thr
        self.ransc_max_iter=ransc_max_iter

    def _img2gray(self, img):
        if len(img.shape) == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

    def __call__(self, img0, img1,debug=""):
        img0 = self._img1gray(img0)
        img1 = self._img1gray(img1)
        img0 = torch.from_numpy(img0)[None][None].cuda() / 255.
        img1 = torch.from_numpy(img1)[None][None].cuda() / 255.

        batch = {'image0': img0, 'image1': img1}
        with torch.no_grad():
            self.model(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()

        idx=np.where(mconf>self.thr)
        mconf=mconf[idx]
        mkpts0=mkpts0[idx]
        mkpts1=mkpts1[idx]

        debug_info=None
        if mkpts0.shape[0] < 4 or mkpts1.shape[0] < 4:
            return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            dtype=np.float), False,debug_info

        H, Mask = cv2.findHomography(mkpts0[:, :2],
                                    mkpts1[:, :2],
                                    self.ransc_method,
                                    self.ransc_thr,
                                    maxIters=self.ransc_max_iter)
        Mask=np.squeeze(Mask)
        if debug:

            kp0_true_matched=mkpts0[Mask.astype(bool),:2]
            kp1_true_matched=mkpts1[Mask.astype(bool),:2]
            kp0_fake_matched=mkpts0[~Mask.astype(bool),:2]
            kp1_fake_matched=mkpts1[~Mask.astype(bool),:2]

            debug_info=DebugInfo(kp0_fake_matched,kp1_fake_matched,kp0_true_matched,kp1_true_matched)

        if H is None:
            return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            dtype=np.float), False,debug_info
        else:
            return H, True,debug_info