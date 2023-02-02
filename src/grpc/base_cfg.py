# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-02-02 16:40:37
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-02-02 16:42:41
@FilePath: /LoFTR/src/grpc/base_cfg.py
@Description:
'''
import json
import yaml

param = dict(grpc=dict(host='127.0.0.1',
                       port='8001',
                       max_workers=10,
                       max_send_message_length=100 * 1024 * 1024,
                       max_receive_message_length=100 * 1024 * 1024),
             loftr=dict(ckpt_path="",device="cuda:0",thr=0.5,ransc_method="USAC_MAGSAC",ransc_thr=3,ransc_max_iter=2000,
))


def _update(dic1: dict, dic2: dict):
    """使用dic2 来递归更新 dic1
        # NOTE:
        1. dic1 本体是会被更改的!!!
        2. python 本身没有做尾递归优化的,dict深度超大时候可能爆栈
    """
    for k, v in dic2.items():
        if k.endswith('args') and v is None:
            dic2[k] = {}
        if k in dic1:
            if isinstance(v, dict) and isinstance(dic1[k], dict):
                _update(dic1[k], dic2[k])
            else:
                dic1[k] = dic2[k]
        else:
            dic1[k] = dic2[k]


def _merge_yaml(yaml_path: str):
    global param
    with open(yaml_path, 'r') as fr:
        content_dict = yaml.load(fr, yaml.FullLoader)
    _update(param, content_dict)


def _merge_json(json_path: str):
    global param
    with open(json_path, 'r') as fr:
        content_dict = json.load(fr)
    _update(param, content_dict)


def merge_param(file_path: str):
    """按照用户传入的配置文件更新基本设置
    """
    cfg_ext = file_path.split('.')[-1]
    func_name = '_merge_' + cfg_ext
    if func_name not in globals():
        raise ValueError('{} is not support'.format(cfg_ext))
    else:
        globals()[func_name](file_path)

