# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-02-03 09:56:39
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-02-03 10:12:32
@FilePath: /LoFTR/src/grpc/debug_tools.py
@Description:
'''
import os
from typing import Optional
import math
from copy import deepcopy
import random

import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np
import torch
from torchvision.utils import make_grid


COLOR_DICT=mcolors.CSS4_COLORS

def get_color():
    return random.sample(COLOR_DICT.keys(),1)[0]

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range *255


def show_img(img_ori,text:Optional[str]=None,cvreader:bool=True,delay:float=0):
    img=deepcopy(img_ori)
    if isinstance(img, list) or isinstance(img, tuple):
        img_num = len(img)
        row_n = math.ceil(math.sqrt(img_num))
        col_n = max(math.ceil(img_num / row_n), 1)
        fig, axs = plt.subplots(row_n, col_n, figsize=(15 * row_n, 15 * col_n))
        for idx, img_ in enumerate(img):
            if isinstance(img_,torch.Tensor) or isinstance(img_,np.ndarray):
                img_=show_tensor(img_,cvreader)
            if 2 == len(axs.shape):
                axs[idx % row_n][idx // row_n].imshow(img_)
                axs[idx % row_n][idx // row_n].set_title(str(idx))
            else:
                axs[idx % row_n].imshow(img_)
                axs[idx % row_n].set_title(str(idx))
        if text:
            plt.text(0,0,text,fontsize=15)
        if delay <=0:
            plt.show()
        else:
            plt.draw()
            plt.pause(delay)
            plt.close()
    elif isinstance(img,torch.Tensor) or isinstance(img,np.ndarray):
        img=show_tensor(img,cvreader)
        plt.imshow(img)
        if text:
            plt.text(0,0,text,fontsize=15)
        if delay <=0:
            plt.show()
        else:
            plt.draw()
            plt.pause(delay)
            plt.close()
    else:
        if hasattr(img,'show'):
            img.show()


def show_tensor(img,cvreader):
    if len(img.shape) == 4 and img.shape[0] !=1:
        if isinstance(img,np.ndarray):
            img=torch.Tensor(img)
            img=make_grid(img)
        else:
            img=make_grid(img)
        img: np.ndarray = img.detach().cpu().numpy().squeeze()
    if isinstance(img,torch.Tensor):
        img=img.detach().cpu().numpy().squeeze()
    if 1==img.max():
        img=img.astype(np.float)
    if img.shape[0] == 1 or img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.min() <0 or img.max() >255:
        img=normalization(img)
        print("img have norm")
        if img.shape[-1] == 3:
            img=img.astype(np.uint8)
    if cvreader and len(img.shape)==3:
        img=img[:,:,::-1]
    return img

def draw_point(axs,kp:np.ndarray,color:str):
    kp=kp.astype(int)
    axs.plot(kp[:,0],kp[:,1],'o',color=color)

def plot_kp(debug_info,show_flag=('fake','true','no_match',"vis"),debug_save=""):
    fig, axs = plt.subplots(2, 1)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].imshow(debug_info.imgA[:,:,::-1])

    axs[1].set_yticks([])
    axs[1].set_yticks([])
    axs[1].imshow(debug_info.imgB[:,:,::-1])

    if "fake" in show_flag:
        draw_point(axs[0],debug_info.pts_info.kp0_fake_match,'y')
        draw_point(axs[1],debug_info.pts_info.kp1_fake_match,'y')
        for kp_a,kp_b in zip(debug_info.pts_info.kp0_fake_match,debug_info.pts_info.kp1_fake_match):
            color=get_color()
            con =  patches.ConnectionPatch(
                xyA=kp_a, coordsA=axs[0].transData,
                xyB=kp_b, coordsB=axs[1].transData,
                arrowstyle="-", shrinkB=5,color=color)
            axs[1].add_artist(con)

    if "true" in show_flag:
        draw_point(axs[0],debug_info.pts_info.kp0_true_match,'g')
        draw_point(axs[1],debug_info.pts_info.kp1_true_match,'g')
        for kp_a,kp_b in zip(debug_info.pts_info.kp0_true_match,debug_info.pts_info.kp1_true_match):
            color=get_color()
            con =  patches.ConnectionPatch(
                xyA=kp_a, coordsA=axs[0].transData,
                xyB=kp_b, coordsB=axs[1].transData,
                arrowstyle="-", shrinkB=5,color=color)
            axs[1].add_artist(con)

    if "vis" in show_flag:
        fig2=plt.figure()
        axe2=fig2.gca()
        fix_img=generate_transform_pic((debug_info.imgB.shape[1],debug_info.imgB.shape[0]),debug_info.imgA,debug_info.H)
        final_img=cv2.addWeighted(debug_info.imgB,0.6,fix_img,0.4,0)
        axe2.imshow(final_img[:,:,::-1])


    if isinstance(debug_save,str):
        save_path,ext=os.path.splitext(debug_save)
        fig.savefig(save_path+"nft.jpg",dpi=400,bbox_inches='tight')
        fig2.savefig(save_path+"vis.jpg",dpi=400,bbox_inches='tight')
        plt.clf()
        plt.close("all")
    else:
        plt.clf()
        plt.show()

def convertpt2matrix(pts):
    if isinstance(pts,list):
        pts=np.array([x+[1] for x in pts])
    if pts.shape[-1]==2:
        expend_m=np.array([[1],[1],[1],[1]])
        pts=np.concatenate((pts,expend_m),axis=1)
    return pts

def generate_transform_pic(imgb_shape,imgA,H):
    imgA_warp = cv2.warpPerspective(imgA,
                                    H, imgb_shape,
                                    flags=cv2.INTER_LINEAR)

    k_pts=[[0,0],[0,imgA.shape[0]-1],[imgA.shape[1]-1,imgA.shape[0]-1],[imgA.shape[1]-1,0]]

    kptm=convertpt2matrix(k_pts)
    warp_kpm=np.matmul(H,kptm.T)
    warp_kpm=(warp_kpm/warp_kpm[2,:])[:2,:].astype(int).T.reshape((-1,1,2))
    # FIXME:这里可能有问题,没做越界
    cv2.polylines(imgA_warp,[warp_kpm],True,(0,0,255),thickness=5)
    return imgA_warp


