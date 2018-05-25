#!/usr/bin/python
# encoding: utf-8
# coding:utf-8


from __future__ import print_function
import argparse
import random

import lmdb
import logging
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import time
import os
import utils
import dataset
from glob import glob
import sys
import gc

sys.path.append("..")

from rookie_utils import mod_config
from rookie_utils.Logger import Logger

import models.crnn as crnn

parser = argparse.ArgumentParser()
parser.add_argument('--trainPath', required=False, help='path to lmdb dataset')
parser.add_argument('--valPath', required=False, help='path to lmdb dataset')
# parser.add_argument('--valroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--ds', required=False, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=100, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=100, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=800, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', default=True,
                    help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
print(opt)

'''
训练流程：
1、将create_dataset 生成的lmdb数据库放在datasets目录下的自己文件夹下
2、本程序将默认读取datasets目录下的每个文件夹中的train和val，分别作为train和val的数据集
3、程序会整个所有的train到tmpLmdb目录下的data.lmdb,该数据每次启动会删除并重建
4、程序会读取生成结果目录( 默认expr )下的网络，重新加载，继续训练
5、workers 指定程序训练过程中生成的进程数，越多占用资源越多
'''

# 日志打印
# 设置上层【项目根目录】为配置文件 所在目录
mod_config.setPath("../server/")
# 项目目录
project_path = mod_config.getConfig("project", "path")
if project_path is None or project_path == '':
    server_path = os.path.split(os.path.realpath(__file__))[0]
    project_path = os.path.dirname(server_path)
# 日志输出
log_path = project_path + mod_config.getConfig("logger", "file")
if not os.path.exists(os.path.dirname(log_path)):
    os.makedirs(os.path.dirname(log_path))
logger = Logger(log_path, logging.INFO, logging.INFO)


def print_msg(msg):
    logger.info(msg)


# 训练结果存储目录
if opt.experiment is None:
    opt.experiment = 'expr'

os.system('mkdir {0}'.format(opt.experiment))

file_path = os.path.dirname(os.path.realpath(__file__))

opt.manualSeed = random.randint(1, 10000)  # fix seed
print_msg("Random Seed: {0}".format(opt.manualSeed))
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print_msg("WARNING: You have a CUDA device, so you should probably run with --cuda")

# 训练数据集
train_loader_list = []
# 验证数据集
val_data_list = []

dataset_dir = opt.trainPath
if dataset_dir is None:
    dataset_dir = file_path + '/splitDB'


def addOneTrain(list, path):
    one_dataset = dataset.lmdbDataset(root=path)
    assert one_dataset
    if opt.random_sample:
        sampler = dataset.randomSequentialSampler(one_dataset, opt.batchSize)
    else:
        sampler = None
    one_loader = torch.utils.data.DataLoader(
        one_dataset, batch_size=opt.batchSize,
        shuffle=True, sampler=sampler,
        num_workers=int(opt.workers),
        collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
    splits = path.split('/')
    obj = {
        'loader': one_loader,
        'flag': splits[-1] if splits[-1] != 'train' else splits[-2]
    }
    list.append(obj)


# 初始化加载 训练数据集
def initTrainDataLoader():
    if os.path.exists(dataset_dir + "/data.mdb"):
        addOneTrain(train_loader_list, dataset_dir)
    else:
        fs = os.listdir(dataset_dir)
        fs.sort()
        for one in fs:
            if not os.path.exists(dataset_dir + "/" + one + "/data.mdb"):
                continue
            addOneTrain(train_loader_list, dataset_dir + "/" + one)
    print("加载了{}个训练集".format(len(train_loader_list)))


# 加载训练数据集
initTrainDataLoader()

# 字符集长度
nclass = len(opt.alphabet) + 1

nc = 1

# 目标数据集的 字符-映射 转换工具
converter = utils.strLabelConverter(opt.alphabet)
# CTCLoss
criterion = CTCLoss()

# if opt.crnn != '':
#     print('loading pretrained model from %s' % opt.crnn)
#     crnn = torch.nn.DataParallel(crnn)
#     crnn.load_state_dict(torch.load(opt.crnn))
# print(crnn)

# 三个张量 分别存储 图片数据、字符串、字符数
image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# 变量的计算工具
# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)

# epochs 迭代训练多少次
for epoch in range(opt.niter):

    fileIndex = 0

    while fileIndex < len(train_loader_list):
        # 本次要训练的模型是哪个
        train_obj = train_loader_list[fileIndex]
        train_loader = train_obj['loader']
        flag = train_obj['flag']
        train_iter = iter(train_loader)
        fileIndex += 1

        i = 0
        while i < len(train_loader):
            i += 1
            train_iter.next()
            print_msg("step:{}".format(i))
            # try:
            #     del train_iter
            # except BaseException as delEx:
            #     print_msg("EX:" + delEx.message + "_" + str(delEx))
            #     print(delEx)
            # finally:
            #     print_msg("一个训练文件结束")
            #     os.popen('sync && echo 3 > /proc/sys/vm/drop_caches')
            #     gc.collect()
# except BaseException as ex:
#     print_msg("EX:" + ex.message + "_" + str(ex))
#     print (ex)
#
# finally:
#     print_msg("Game Over")
