#!/usr/bin/python
# encoding: utf-8

from __future__ import print_function
from __future__ import print_function
from __future__ import print_function

import argparse
import logging

import os
import sys


sys.path.append("../..")
from rookie_utils.Logger import Logger
from crnn_pytorch import dataset

parser = argparse.ArgumentParser()
parser.add_argument('--result', required=True, help='path to result')
parser.add_argument('--reset', required=True, default=False, help='del the exist lmdb')
parser.add_argument('--srcPath', required=True, help='path to resource')
opt = parser.parse_args()
print(opt)


# lmdb 整合
def intergrate(result_lmdb, reset, srcPath, logger=None):
    # 用于整合的lmdb
    src_list = []

    if not os.path.exists(srcPath):
        logger("数据源目录不存在，终止合并")
        return
    if not os.path.exists(result_lmdb):
        os.mkdir(result_lmdb)
    elif os.path.exists(result_lmdb+"/data.mdb") and reset:
        os.remove(result_lmdb+"/data.mdb")
        os.remove(result_lmdb+"/lock.mdb")

    # 单目录整合
    if os.path.exists(srcPath + "/data.mdb"):
        src_list.append(srcPath)
    else:
        # 多目录整合
        paths = os.listdir(srcPath)
        for p in paths:
            if not os.path.isdir(srcPath + "/" + p):
                continue
            path = srcPath + "/" + p
            if os.path.exists(path + "/data.mdb"):
                src_list.append(path)
            elif os.path.exists(path + "/train/data.mdb"):
                src_list.append(path + "/train")

    for path in src_list:
        sta = dataset.merge_lmdb(result_lmdb, path, max_size=-1, logger=logger)
        logger(sta)



# 日志输出
log_path = "../../logs/integrate.log"
if not os.path.exists(os.path.dirname(log_path)):
    os.makedirs(os.path.dirname(log_path))
logger = Logger(log_path, logging.INFO, logging.INFO)


def print_msg(msg):
    logger.info(msg)


if __name__ == '__main__':
    if (opt.result is None or opt.result == '') and (opt.srcPath is None or opt.srcPath == ''):
        raise Exception('imagePath and iamgeDirPath must not to be both blank')
    print_msg("将要吧{}的lmdb附加到{}".format(opt.srcPath, opt.result))
    intergrate(opt.result, opt.reset, opt.srcPath, print_msg)
