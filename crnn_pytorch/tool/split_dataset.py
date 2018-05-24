#!/usr/bin/python
# encoding: utf-8
# coding:utf-8

import argparse
import logging
import os
import lmdb  # install lmdb by "pip install lmdb"
import sys

sys.path.append("../..")
sys.path.append("..")
from rookie_utils.Logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument('--src', required=True, help='resource lmdbs to split ')
parser.add_argument('--out', required=True, help='path to save split results')
parser.add_argument('--count', type=int, required=False, default=1000000, help='file num each out contains')
opt = parser.parse_args()
print(opt)

'''
分割流程
src为单个lmdb目录或者多个lmdb目录的所属目录
out为输出的分割结果坐在目录
count是多少个样本为一个文件
多个输入的话，会遍历着输入
'''

# 日志输出
log_path = "../../logs/split.log"
if not os.path.exists(os.path.dirname(log_path)):
    os.makedirs(os.path.dirname(log_path))
logger = Logger(log_path, logging.INFO, logging.INFO)


def print_msg(msg):
    logger.info(msg)


def addOne(list, path):
    env = lmdb.open(path)
    txn = env.begin()
    size = txn.get('num-samples')
    if size is None or size == '':
        size = '0'

    obj = {
        'env': env,
        'index': 0,
        'size': int(size),
        'txn': txn
    }
    list.append(obj)


def writeOne(txnFrom, fromIndex, txnWrite, writeIndex):
    image_key = "image-%09d" % (fromIndex)
    lable_key = "label-%09d" % (fromIndex)
    new_image_key = "image-%09d" % (writeIndex)
    new_lable_key = "label-%09d" % (writeIndex)
    txnWrite.put(new_image_key, txnFrom.get(image_key))
    txnWrite.put(new_lable_key, txnFrom.get(lable_key))


#
def split(src, out, count):
    # 要输出的位置
    if os.path.exists(out):
        if len(os.listdir(out)) != 0:
            print_msg("输出目录已存在，且内含有文件,终止，请选择一个空目录")
            sys.exit(0)

    list_src_obj = []
    # 直接存在数据库文件，则分割是单文件分割
    if os.path.exists(src + "/data.mdb"):
        addOne(list_src_obj, src)
    else:
        list_path = os.listdir(src)
        for path in list_path:
            if not os.path.exists(src + "/" + path + "/train/data.mdb"):
                continue
            # 添加
            addOne(list_src_obj, src + "/" + path + "/train")

    print_msg("要分割的数据源数量:{}".format(len(list_src_obj)))

    # 处理输出
    if not os.path.exists(out):
        os.mkdir(out)
    out_index = 1
    out_name = out + "/" + ('splitdata%03d' % out_index)
    out_env = lmdb.open(out_name, map_size=1099511627776)
    out_txn = out_env.begin(write=True)

    # 读取总进度，每个1代表N个文件的
    totol_index = 1
    # 写入进度，标识当前文件写入的位置
    cur_index = 0
    keepRuning = True
    while keepRuning:
        # 每个里面取一个，写进去
        keepRuning = False

        for obj in list_src_obj:
            if totol_index <= obj['size']:
                keepRuning = True
                txn = obj['txn']
                cur_index += 1
                writeOne(txn, totol_index, out_txn, cur_index)
        if not keepRuning:
            out_txn.put('num-samples', str(cur_index))
            out_txn.commit()
            out_env.close()
            break
        # 部分提交
        if cur_index % 1000 == 0:
            out_txn.put('num-samples', str(cur_index))
            out_txn.commit()
            out_txn = out_env.begin(write=True)
            print_msg("write into {},size:{}".format(out_name, cur_index))
        # 这个写的数据量足够了，开始写下一个文件
        if cur_index >= (count - len(list_src_obj)):
            if cur_index % 1000 != 0:
                out_txn.put('num-samples', str(cur_index))
                print_msg("write into {},size:{}".format(out_name, cur_index))
                out_txn.commit()
            out_env.close()
            out_index += 1
            out_name = out + "/" + ('splitdata%03d' % out_index)
            out_env = lmdb.open(out_name, map_size=1099511627776)
            out_txn = out_env.begin(write=True)
            cur_index = 1
        # 读取位置前进一步
        totol_index += 1

    print_msg("GameOver")


if __name__ == '__main__':
    count = opt.count
split(opt.src, opt.out, count)
