#!/usr/bin/python
# encoding: utf-8
# coding:utf-8


from __future__ import print_function
import argparse
import random

import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np

import dataset
import sys

sys.path.append("..")

parser = argparse.ArgumentParser()
parser.add_argument('--lmdbPath', required=False, help='path to lmdb dataset')
# parser.add_argument('--valroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
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

opt.manualSeed = random.randint(1, 10000)  # fix seed

random.seed(opt.manualSeed)

np.random.seed(opt.manualSeed)

torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

# 训练数据集
train_data_list = []
# 验证数据集
val_data_list = []

print("开始加载临时数据库中的全部数据")
lmdb_path = "/home/rookie/work/githubWork/ocr_crnn/crnn_pytorch/ydsc4IO"

train_dataset = dataset.lmdbDataset(root=lmdb_path)
dataset.lmdbDataset(root=lmdb_path, transform=dataset.resizeNormalize((100, 32)))
assert train_dataset
print("加载临时数据库 成功")

if opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None

data_loader = torch.utils.data.DataLoader(
    train_dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
val_iter = iter(data_loader)
count = len(data_loader)
for i in range(count):
    val_iter.next()
print('end')
