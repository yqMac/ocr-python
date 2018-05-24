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
parser.add_argument('--lmdbPath', required=False, help='path to lmdb dataset')
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

dataset_dir = opt.lmdbPath
if dataset_dir is None:
    dataset_dir = file_path + '/splitDB'

sampler = None


def addOneTrain(list, path):
    one_dataset = dataset.lmdbDataset(root=path)
    assert one_dataset
    if opt.random_sample:
        sampler = dataset.randomSequentialSampler(one_dataset, opt.batchSize)
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
        for one in fs:
            if not os.path.exists(dataset_dir + "/" + one + "/data.mdb"):
                continue
            addOneTrain(train_loader_list, dataset_dir + "/" + one)
    print("加载了{}个训练集:{}".format(len(train_loader_list)))


# 初始化加载 验证数据集
def initValDataSets():
    fs = os.listdir(dataset_dir)
    index = 0
    list_name = []

    for one in fs:
        root_path = dataset_dir + "/" + one + "/val"
        if not os.path.exists(root_path):
            continue
        # print("添加校验数据集:{}".format(root_path))
        one_dataset = dataset.lmdbDataset(root=root_path, transform=dataset.resizeNormalize((100, 32)))

        # one_loader = torch.utils.data.DataLoader(one_dataset, shuffle=True, batch_size=opt.batchSize,
        #                                          num_workers=int(opt.workers))
        val_data = {
            "dir": one,
            "dataset": one_dataset,
            # "loader": one_loader,
            "index": index
        }
        index += 1
        val_data_list.append(val_data)
        list_name.append(one)
    print_msg("加载了{}个验证集:{}".format(len(list_name), list_name))


# 加载训练数据集
initTrainDataLoader()

# 加载校验数据集
initValDataSets()

# 字符集长度
nclass = len(opt.alphabet) + 1

nc = 1

# 目标数据集的 字符-映射 转换工具
converter = utils.strLabelConverter(opt.alphabet)
# CTCLoss
criterion = CTCLoss()


# 自定义 权重初始化 ，被crnn调用
# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
crnn.apply(weights_init)

# 继续训练
crnnPath = opt.crnn
if crnnPath is None or crnnPath == '':
    crnnPath = file_path + '/expr'
if crnnPath is not None:
    pths = os.listdir(crnnPath)
    # 解决了加载失败的问题，不需要找倒数第二个了，找最新的就行
    if len(pths) > 0:
        pths.sort()
        if pths[len(pths) - 1].endswith(".pth"):
            continue_path = crnnPath + "/" + pths[len(pths) - 1]
            print_msg("从上次文件继续训练:{}".format(continue_path))
            crnn = torch.nn.DataParallel(crnn)
            state_dict = torch.load(continue_path)
            try:
                crnn.load_state_dict(state_dict)
            except Exception as ex:
                print_msg("加载时发生异常{0}，开始尝试使用自定义dict".format(ex.message))
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # load params
                crnn.load_state_dict(new_state_dict)
        else:
            print_msg("你这不符合格式啊:{}".format(pths[0]))

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


def val(crnn, val_data_list_param, criterion, max_iter=100):
    # print('开始校验准确性')
    for p in crnn.parameters():
        p.requires_grad = False
    crnn.eval()
    i = 0
    all_Count = 0
    correct_Count = 0
    while i < len(val_data_list_param):
        val_data = val_data_list_param[i]
        # datasetOne = datasetList[i]
        # data_loader = torch.utils.data.DataLoader(
        #     datasetOne, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
        data_set = val_data['dataset']
        data_loader = torch.utils.data.DataLoader(
            data_set, shuffle=True, batch_size=opt.batchSize, num_workers=1)
        i += 1

        # print("验证进度:{}/{},当前Flag:{}".format(i, len(val_data_list_param), val_data['dir']))

        val_iter = iter(data_loader)
        one_index = 0
        one_correct = 0
        loss_avg = utils.averager()
        # 检测所用的图片数量
        max_iter = min(max_iter, len(data_loader))
        # 检测的总数量增加
        all_Count += max_iter * opt.batchSize

        for one_index in range(max_iter):

            data = val_iter.next()
            one_index += 1
            cpu_images, cpu_texts = data
            batch_size = cpu_images.size(0)
            utils.loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts)
            utils.loadData(text, t)
            utils.loadData(length, l)

            preds = crnn(image)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            # cost = criterion(preds, text, preds_size, length)
            cost = criterion(preds, text, preds_size, length) / batch_size

            loss_avg.add(cost)
            _, preds = preds.max(2, keepdim=True)
            preds = preds.squeeze(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, cpu_texts):
                if pred == target.lower():
                    # 两个成功数量都加1
                    one_correct += 1
                    correct_Count += 1
        del val_iter
        raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
        accuracy = one_correct / float(max_iter * opt.batchSize)
        if accuracy < 0.95:
            for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
                print_msg('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

        print_msg('验证 %-3d/%d,Loss: %f,Flag: [%-15s] 的成功率: %f' % (
            i, len(val_data_list_param), loss_avg.val(), val_data['dir'], accuracy))

    accuracy = correct_Count / float(all_Count)
    print_msg('总的成功率: %f ,总验证文件数: %d ' % (accuracy, all_Count))
    return accuracy


# 训练一个Batch
def trainBatch(crnn, iter, criterion, optimizer):
    # 取一个Batch的数据集
    data = iter.next()
    # 区分图片 和 标签
    cpu_images, cpu_texts = data

    batch_size = cpu_images.size(0)
    # 图片数据加载到张量
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    # 标签数据加载到张量
    utils.loadData(text, t)
    # 长度数据加载到张量
    utils.loadData(length, l)

    # 执行forward
    preds = crnn(image)

    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size

    crnn.zero_grad()
    cost.backward()
    optimizer.step()

    return cost


def keep_only_models(n=10):
    model_files = sorted(glob(opt.experiment + '/{0}*'.format("netCRNN")))
    models_to_delete = model_files[:-n]
    for model_file in models_to_delete:
        print_msg('remove other model:{}'.format(model_file))
        os.remove(model_file)


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
        one_train_step = 0
        # 取合适的存储时机
        saveInterval = min(opt.saveInterval, len(train_loader))

        i = 0
        while i < len(train_loader):
            for p in crnn.parameters():
                p.requires_grad = True
            crnn.train()

            cost = trainBatch(crnn, train_iter, criterion, optimizer)
            loss_avg.add(cost)
            i += 1

            # 多少次batch显示一次进度
            if i % opt.displayInterval == 0:
                print_msg(
                    'epoch:[%-5d/%d],flag:[%-10s],step: [%-4d/%d], Loss: %f' % (
                        epoch, opt.niter, flag, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

            # 检查点:检查成功率,存储model，
            if i % saveInterval == 0 or i == len(train_loader):
                certVal = val(crnn, val_data_list, criterion)
                time_format = time.strftime('%Y%m%d_%H%M%S')
                print_msg("save model: {0}/netCRNN_{1}_{2}.pth".format(opt.experiment, time_format, int(certVal * 100)))
                torch.save(crnn.state_dict(),
                           '{0}/netCRNN_{1}_{2}.pth'.format(opt.experiment, time_format, int(certVal * 100)))
                keep_only_models()
                gc.collect()
        del train_iter
        os.popen('sync && echo 3 > /proc/sys/vm/drop_caches')
        break
