#!/usr/bin/python
# encoding: utf-8

from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np


class lmdbDataset(Dataset):
    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key))

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)


# read lmdb2 data then write into lmdb1
def merge_lmdb(result_lmdb, lmdb2):
    # env代表Environment, txn代表Transaction
    print('Merge start!')
    # 打开lmdb文件，读模式
    env_2 = lmdb.open(lmdb2)
    # 创建事务
    txn_2 = env_2.begin()

    count_2 = txn_2.get('num-samples')

    # 打开数据库
    database_2 = txn_2.cursor()

    # 打开lmdb文件，写模式，
    env_3 = lmdb.open(result_lmdb, map_size=int(1e12))
    txn_3 = env_3.begin(write=True)
    count_3 = txn_3.get('num-samples')
    if count_3 is None:
        count_3 = '0'
    count_2 = int(count_2)
    count_3 = int(count_3)
    count_total = count_2 + count_3
    count = 0
    # 遍历数据库
    for (key, value) in database_2:
        new_key = str(key)
        if new_key.startswith("image-"):
            new_key = new_key.replace("image-", "")
            new_key = "image-%09d" % (count_3 + int(new_key))
        elif new_key.startswith("label-"):
            new_key = new_key.replace("label-", "")
            new_key = "label-%09d" % (count_3 + int(new_key))
        else:
            continue
        if count == 0:
            print("first change new_key: {} ".format(new_key))
        txn_3.put(new_key, value)
        count += 1
        if count % 1000 == 0:
            print("Merge: {}".format(count))
            txn_3.commit()
            txn_3 = env_3.begin(write=True)

    if count % 1000 != 0:
        txn_3.commit()
        txn_3 = env_3.begin(write=True)

    # 更新大小
    txn_3.put("num-samples", str(count_total))
    # 输出结果lmdb的状态信息，可以看到数据是否合并成功
    print(env_3.stat())
    # 关闭lmdb
    env_2.close()
    env_3.close()
    print('Merge success! count: {}'.format(count))


class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):
    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
