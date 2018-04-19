#!/usr/bin/python
# encoding: utf-8
# coding:utf-8


from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
from glob import glob

import models.crnn as crnn

parser = argparse.ArgumentParser()
parser.add_argument('--lmdbPath', required=True, help='path to lmdb dataset')
# parser.add_argument('--valroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
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
parser.add_argument('--saveInterval', type=int, default=100, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', default=True,
                    help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
print(opt)

# 训练结果存储目录
if opt.experiment is None:
    opt.experiment = 'expr'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# 训练的数据集集合
train_dataset_list = []
# 训练的加载器
train_loader_list = []

# 验证的数据集
val_dataset_list = []
# 验证的加载器
val_loader_list = []

dataset_dir = 'datasets'


# 初始化加载 训练数据集
def initTrainDataSets():
    trains_dir = dataset_dir + "/trains"
    fs = os.listdir(trains_dir)
    for one in fs:
        if not one.endswith(".mdb"):
            continue
        one_dataset = dataset.lmdbDataset(root=one)
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
        train_dataset_list.append(one_dataset)
        train_loader_list.append(one_loader)


# 初始化加载 验证数据集
def initValDataSets():
    vals_dir = dataset_dir + "/vals"
    fs = os.listdir(vals_dir)
    for one in fs:
        if not one.endswith(".mdb"):
            continue
        one_dataset = dataset.lmdbDataset(root=one)
        # assert one_dataset
        # if opt.random_sample:
        #     sampler = dataset.randomSequentialSampler(one_dataset, opt.batchSize)
        # else:
        #     sampler = None
        # one_loader = torch.utils.data.DataLoader(
        #     one_dataset, batch_size=opt.batchSize,
        #     shuffle=True, sampler=sampler,
        #     num_workers=int(opt.workers),
        #     collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
        val_dataset_list.append(one_dataset)
        # train_loader_list.append(one_loader)


# 字符集长度
nclass = len(opt.alphabet) + 1

nc = 1

# 目标数据集的 字符-映射 转换工具
converter = utils.strLabelConverter(opt.alphabet)
# CTCLoss
criterion = CTCLoss()



# 加载训练和验证集的目录
trainroot = opt.lmdbPath + "/train"
valroot = opt.lmdbPath + "/val"

train_dataset = dataset.lmdbDataset(root=trainroot)

# 断言不为null，否则抛异常
assert train_dataset
if opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
test_dataset = dataset.lmdbDataset(
    root=valroot, transform=dataset.resizeNormalize((100, 32)))


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
if opt.crnn != '':
    print('loading pretrained model from %s' % opt.crnn)
    crnn = torch.nn.DataParallel(crnn)
    crnn.load_state_dict(torch.load(opt.crnn))
print(crnn)

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


def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2, keepdim=True)
        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


# 训练一个Batch
def trainBatch(crnn, criterion, optimizer):

    # 取一个Batch的数据集
    data = train_iter.next()
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
        print(model_file)
        os.remove(model_file)


# epochs 迭代训练多少次
for epoch in range(opt.niter):
    # loader的指针
    train_iter = iter(train_loader)
    i = 0
    # 一次迭代
    while i < len(train_loader):
        # 第几次迭代的第几个batch
        print("epoch: {}, step: {}".format(epoch, i))
        # 所有变量都要求梯度
        for p in crnn.parameters():
            p.requires_grad = True

        # 设置为训练模式
        crnn.train()
        # 训练一个Batch
        cost = trainBatch(crnn, criterion, optimizer)
        loss_avg.add(cost)
        i += 1

        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.niter, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        # do checkpointing
        if i % opt.saveInterval == 0:
            val(crnn, test_dataset, criterion)
            # print("save model: {0}/netCRNN_{1}_{2}.pth".format(opt.experiment, epoch, i))
            print("save model: {0}/netCRNN.pth".format(opt.experiment))
            # torch.save(crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.experiment, epoch, i))
            torch.save(crnn.state_dict(), '{0}/netCRNN.pth'.format(opt.experiment))
            keep_only_models()

