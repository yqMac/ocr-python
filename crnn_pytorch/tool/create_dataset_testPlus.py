#!/usr/bin/python
# encoding: utf-8
# coding:utf-8

import argparse
import os
import re
from glob import glob
import random

import cv2
import lmdb  # install lmdb by "pip install lmdb"
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--imagePath', required=False, help='path to image')
parser.add_argument('--imageDirPath', required=False, help='path to image path dirs ')
parser.add_argument('--lmdbPath', required=False, help='path to lmdb')
parser.add_argument('--head', required=False, help='file name pre to save')
opt = parser.parse_args()
print(opt)


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, outputHead, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    split the imagesList to ten parts, nine for train, one for val
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        checkValid    : if true, check the validity of every image
    """
    # assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    print(nSamples)
    train_size = int(round(round(nSamples / 10000.0, 1) * 9, 0) * 1000)
    print(train_size)
    val_size = nSamples - train_size
    print(val_size)
    if outputPath is None or outputPath == "":
        file_path = os.path.dirname(os.path.realpath(__file__))
        crnn_path = os.path.dirname(file_path)
        datasets_path = crnn_path + '/datasets'
        outputPath = datasets_path
    train_lmdb_path = outputPath + "/" + outputHead + "/train"
    val_lmdb_path = outputPath + "/" + outputHead + "/val"
    if not os.path.exists(train_lmdb_path):
        os.makedirs(train_lmdb_path)
    if not os.path.exists(val_lmdb_path):
        os.makedirs(val_lmdb_path)

    env_train = lmdb.open(train_lmdb_path, map_size=1099511627776)
    env_val = lmdb.open(val_lmdb_path, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        try:
            match = re.compile("^.*_(.*)\..+$").match(imagePath.split("/")[-1])
            # match = re.compile("^(.*)\..+$").match(imagePath.split("/")[-1])
            label = match.group(1)
            if not os.path.exists(imagePath):
                print('%s does not exist' % imagePath)
                continue
            with open(imagePath, 'r') as f:
                imageBin = f.read()
            if checkValid:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue

            imageKey = 'image-%09d' % cnt
            labelKey = 'label-%09d' % cnt
            cache[imageKey] = imageBin
            cache[labelKey] = label
            if (i + 1) % 1000 == 0:
                if (i + 1) <= train_size:
                    writeCache(env_train, cache)
                    print('Written train %d / %d' % (cnt, train_size))
                    cache = {}
                    if cnt == train_size:
                        cnt = 0
                else:
                    writeCache(env_val, cache)
                    print('Written val %d / %d' % (cnt, val_size))
                    cache = {}
            cnt += 1
        except Exception as e:
            print("the image {0} is error{1}".format(imagePath, e.message))
    cache['num-samples'] = str(val_size)
    writeCache(env_val, cache)
    print('Created val dataset with %d samples' % val_size)
    cache.clear()
    cache['num-samples'] = str(train_size)
    writeCache(env_train, cache)
    print('Created train dataset with %d samples' % train_size)


if __name__ == '__main__':
    if (opt.imagePath is None or opt.imagePath == '') and (opt.imageDirPath is None or opt.imageDirPath == ''):
        raise Exception('imagePath and iamgeDirPath must not to be both blank')
    if opt.imageDirPath is not None and opt.imageDirPath != '':
        fs = glob(opt.imageDirPath + "/*")
        for file in fs:
            if not os.path.isdir(file):
                print ("不是文件夹,跳过{}".format(file))
                continue
            head = file
            path = opt.imageDirPath + "/" + file
            if os.path.exists(path + "/success"):
                path += "/success"

            paths = glob(path + "/*.*")
            print("初始化加载:dir:{},flag:{}".format(path,head))
            random.shuffle(paths)
            createDataset(opt.lmdbPath, paths, head)
    else:
        paths = glob(opt.imagePath + "/*.*")
        print("split the imagePathList to ten parts, nine for train, one for val")
        random.shuffle(paths)
        createDataset(opt.lmdbPath, paths, opt.head)
