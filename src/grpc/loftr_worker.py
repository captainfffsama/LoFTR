# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2023年 02月 02日 星期四 15:19:14 CST
@Description:
'''
from collections import namedtuple
from typing import Optional, Union

import torch
import cv2
import numpy as np

from src.loftr import LoFTR, default_cfg
from .debug_tools import plot_kp

KeyPointsDebugInfo = namedtuple(
    "KeyPointsDebugInfo",
    ["kp0_fake_match", "kp1_fake_match", "kp0_true_match", "kp1_true_match"])


class DebugInfoCollector(object):

    def __init__(self,
                 imgA: Optional[np.ndarray] = None,
                 imgB: Optional[np.ndarray] = None,
                 pts_info: Optional[KeyPointsDebugInfo] = None,
                 H: Optional[np.ndarray] = None):
        self.imgA = imgA
        self.imgB = imgB
        self.pts_info = pts_info
        self.H = H

    def clean(self):
        self.imgA = None
        self.imgB = None
        self.pts_info = None
        self.H = None

    def __str__(self):
        return "{}".format({
            "imgA": self.imgA,
            "imgB": self.imgB,
            "pts_info": self.pts_info,
            "H": self.H
        })


class LoFTRWorker(object):

    def __init__(self,
                 config,
                 ckpt_path,
                 img_size=(640, 480),
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
        self.thr = thr
        self.ransc_method = getattr(cv2, ransc_method)
        self.ransc_thr = ransc_thr
        self.ransc_max_iter = ransc_max_iter
        self.img_size = img_size

    def _img2gray(self, img):
        if len(img.shape) == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _imgdeal(self, img):
        oh, ow = img.shape[:2]
        img = cv2.resize(img, self.img_size)
        h, w = img.shape[:2]
        fix_matrix = np.array([[w / ow, 0, 0], [0, h / oh, 0], [0, 0, 1]])
        return img, fix_matrix

    def _fix_H(self, fm0, fm1, H):
        return np.linalg.inv(fm0) @ H @ fm1

    def __call__(self,
                 img0,
                 img1,
                 debug: Union[bool, str] = False,
                 debug_show_type: tuple = (
                     "vis",
                     "false",
                     "true",
                 )):
        img0_o, fm0 = self._imgdeal(img0)
        img1_o, fm1 = self._imgdeal(img1)
        img0 = self._img2gray(img0_o)
        img1 = self._img2gray(img1_o)
        img0 = torch.from_numpy(img0)[None][None].cuda() / 255.
        img1 = torch.from_numpy(img1)[None][None].cuda() / 255.

        batch = {'image0': img0, 'image1': img1}
        with torch.no_grad():
            self.model(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()

        idx = np.where(mconf > self.thr)
        mconf = mconf[idx]
        mkpts0 = mkpts0[idx]
        mkpts1 = mkpts1[idx]

        H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float)
        if mkpts0.shape[0] < 4 or mkpts1.shape[0] < 4:
            return self._fix_H(fm0, fm1, H), False

        H, Mask = cv2.findHomography(mkpts0[:, :2],
                                     mkpts1[:, :2],
                                     self.ransc_method,
                                     self.ransc_thr,
                                     maxIters=self.ransc_max_iter)
        Mask = np.squeeze(Mask)
        if debug:

            kp0_true_matched = mkpts0[Mask.astype(bool), :2]
            kp1_true_matched = mkpts1[Mask.astype(bool), :2]
            kp0_fake_matched = mkpts0[~Mask.astype(bool), :2]
            kp1_fake_matched = mkpts1[~Mask.astype(bool), :2]

            kpdi = KeyPointsDebugInfo(kp0_fake_matched, kp1_fake_matched,
                                      kp0_true_matched, kp1_true_matched)

            debug_info = DebugInfoCollector(img0_o, img1_o, kpdi, H)
            plot_kp(debug_info, show_flag=debug_show_type, debug_save=debug)
        if H is None:
            return self._fix_H(fm0, fm1, H), False
        else:
            return self._fix_H(fm0, fm1, H), True
