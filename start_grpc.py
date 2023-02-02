# -*- coding: utf-8 -*-
'''
@Author: captainfffsama
@Date: 2023-02-02 16:38:45
@LastEditors: captainfffsama tuanzhangsama@outlook.com
@LastEditTime: 2023-02-02 16:48:12
@FilePath: /LoFTR/start_grpc.py
@Description:
'''
from concurrent import futures
import sys
from pprint import pprint
import os
import yaml

import grpc

from src.grpc.loftr_pb2_grpc import add_LoftrServicer_to_server
import src.grpc.base_cfg as cfg
from src.grpc.server import LoFTRServer


def start_server(config):
    if not os.path.exists(config):
        raise FileExistsError('{} 不存在'.format(config))
    cfg.merge_param(config)
    args_dict: dict = cfg.param
    pprint(args_dict)

    grpc_args = args_dict['grpc']
    model_args = args_dict['loftr']
    #  最大限制为100M
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=grpc_args['max_workers']),
        options=[('grpc.max_send_message_length',
                  grpc_args['max_send_message_length']),
                 ('grpc.max_receive_message_length',
                  grpc_args['max_receive_message_length'])])

    loftr_server =LoFTRServer(**model_args)
    add_LoftrServicer_to_server(loftr_server,server)
    server.add_insecure_port("{}:{}".format(grpc_args['host'],
                                            grpc_args['port']))
    server.start()
    server.wait_for_termination()


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description="grpc调用loftr,需要配置文件")
    parser.add_argument("-c", "--config", type=str, default="", help="配置文件地址")
    options = parser.parse_args(args)
    if options.config:
        start_server(options.config)

if __name__ == "__main__":
    rc = 1
    try:
        main()
    except Exception as e:
        print('Error: %s' % e, file=sys.stderr)
    sys.exit(rc)