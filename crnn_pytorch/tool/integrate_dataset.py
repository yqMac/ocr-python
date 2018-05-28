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


#
# # read lmdb2 data then write into lmdb1
# def merge_lmdb(result_lmdb, lmdb2, max_size=-1, logger=None):
#     # env代表Environment, txn代表Transaction
#     print('Merge start!')
#     # 打开lmdb文件，读模式
#     env_2 = lmdb.open(lmdb2)
#     # 创建事务
#     txn_2 = env_2.begin()
#
#     count_2 = txn_2.get('num-samples')
#
#     # 打开数据库
#     database_2 = txn_2.cursor()
#
#     # 打开lmdb文件，写模式，
#     env_3 = lmdb.open(result_lmdb, map_size=int(1e12))
#     txn_3 = env_3.begin(write=True)
#     count_3 = txn_3.get('num-samples')
#     if count_3 is None:
#         count_3 = '0'
#     count_2 = int(count_2)
#     count_3 = int(count_3)
#     if max_size == -1 or max_size >= count_2:
#         max_size = count_2
#
#     count_total = max_size + count_3
#     if logger!=None:
#         logger("数据集:{},总:{},处理:{},整合后:{}".format(lmdb2,count_2,max_size,count_total))
#     count = 1
#     # 遍历数据库
#     while count <= max_size:
#         image_key = "image-%09d" % (count)
#         lable_key = "label-%09d" % (count)
#         new_image_key = "image-%09d" % (count_3 + count)
#         new_lable_key = "label-%09d" % (count_3 + count)
#         txn_3.put(new_image_key, txn_2.get(image_key))
#         txn_3.put(new_lable_key, txn_2.get(lable_key))
#         if count % 1000 == 0:
#             print("Merge: {}".format(count))
#             txn_3.commit()
#             txn_3 = env_3.begin(write=True)
#         count += 1
#     if count % 1000 != 0:
#         txn_3.commit()
#         txn_3 = env_3.begin(write=True)
#
#     # 更新大小
#     txn_3.put("num-samples", str(count_total))
#     txn_3.commit()
#     # 输出结果lmdb的状态信息，可以看到数据是否合并成功
#     res = env_3.stat()
#     print(res)
#     # 关闭lmdb
#     env_2.close()
#     env_3.close()
#     print('Merge success! count: {}'.format(count))
#     return res

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
