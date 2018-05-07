#! /usr/bin/env python
# -*- coding: utf-8 -*-

from StringIO import StringIO
import base64
import json
import os
import sys
import tensorflow.contrib.slim as slim
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
import logging
import time
logger = logging.getLogger('Training a chinese write char recognition')
logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

reload(sys)

sys.setdefaultencoding('utf-8')


tf.app.flags.DEFINE_integer('charset_size', 3829, "Choose the first `charset_size` characters only.")
tf.app.flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_string('checkpoint_Path', '../chineseImage/', "don't find checkpoint_path.")
tf.app.flags.DEFINE_string('checkpoint_dir', '', "don't find checkpoint_dir.")
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')
FLAGS = tf.app.flags.FLAGS

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

#根据siteID 分别赋不同的值

def getParameter(siteId):
    if "40001" == siteId:
        FLAGS.image_size = 64
        FLAGS.charset_size = 3777
    # else:
    #     FLAGS.image_size = 64
    #     FLAGS.charset_size = 3772



def build_graph(top_k):
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')  # dropout打开概率
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='train_flag')
    with tf.device('/gpu:5'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training}):
            conv3_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv3_1')
            max_pool_1 = slim.max_pool2d(conv3_1, [2, 2], [2, 2], padding='SAME', scope='pool1')
            conv3_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv3_2')
            max_pool_2 = slim.max_pool2d(conv3_2, [2, 2], [2, 2], padding='SAME', scope='pool2')
            conv3_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3_3')
            max_pool_3 = slim.max_pool2d(conv3_3, [2, 2], [2, 2], padding='SAME', scope='pool3')
            conv3_4 = slim.conv2d(max_pool_3, 512, [3, 3], padding='SAME', scope='conv3_4')
            conv3_5 = slim.conv2d(conv3_4, 512, [3, 3], padding='SAME', scope='conv3_5')
            max_pool_4 = slim.max_pool2d(conv3_5, [2, 2], [2, 2], padding='SAME', scope='pool4')

            flatten = slim.flatten(max_pool_4)
            fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024,
                                       activation_fn=tf.nn.relu, scope='fc1')
            logits = slim.fully_connected(slim.dropout(fc1, keep_prob), FLAGS.charset_size, activation_fn=None,
                                          scope='fc2')
        probabilities = tf.nn.softmax(logits)

        # 返回top k 个预测结果及其概率；返回top K accuracy
        predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
        accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

    return {'images': images,
            'labels': labels,
            'keep_prob': keep_prob,
            'top_k': top_k,
            'is_training': is_training,
            'accuracy_top_k': accuracy_in_top_k,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}


def inference(image, checkpoint_dir, sessMap, grapMap, siteId):
    image_set = []

    # 使用with as 的方式每次都会关闭TensorFlow的session，使用sess = tf.session 的方式需要手动关闭session，如果不关闭下次可供使用
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    # 对每张图进行尺寸标准化和归一化
    temp_image = image.convert('L')
    temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
    temp_image = np.asarray(temp_image) / 255.0
    temp_image = temp_image.reshape([-1, 64, 64, 1])
    image_set.append(temp_image)

    if ckpt and not siteId in sessMap.keys() and not siteId in grapMap.keys():
        graph = build_graph(top_k=1)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)
        sessMap[siteId] = sess
        grapMap[siteId] = graph
    else:
        sess = sessMap.get(siteId)
        graph = grapMap.get(siteId)
    val_list = []
    idx_list = []
    # 预测每一张图

    for item in image_set:
        temp_image = item
        predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                                    feed_dict={graph['images']: temp_image,
                                                            graph['keep_prob']: 1.0,
                                                            graph['is_training']: False})
        val_list.append(predict_val)
        idx_list.append(predict_index)
    return val_list, idx_list


# 获取汉字label映射表,跟句checkpointId找到对应的识别中文对应表
def get_label_dict(label_dir):
    f = open(label_dir, 'r')
    label_dict = pickle.load(f)
    f.close()
    return label_dict

#把图片缩放成指定大小正方形图片
def ImageChangeSize(image, width, height, marginSize):
    image = erzhihua(image, 150)
    image = addBorder(image, marginSize)
    #image.save("/Users/shangzhen/Desktop/jbxx/jbxx2" + str(j) + "_" + str(z) + ".png")
    image = suofangImage(image, width, height)
    #image.save("/Users/shangzhen/Desktop/jbxx/jbxx2" + str(j) + "_" + str(z) + ".png")
    return image

#增加边框
def addBorder(img, length):
    x = img.size[0]
    y = img.size[1]
    #创建长宽为length的黑色图片，
    newImage = Image.new("RGB", (x+length, y+length), (0, 0, 0))
    #newImage.save("/Users/shangzhen/Desktop/jbxx/jbxx2000.png")
    img = img.convert("RGB")
    pixImg = img.load()
    newImage = newImage.convert("RGB")
    pixNewImage = newImage.load()
    # pixdata[x, y] = (255, 255, 255) 给图片指定像素图上颜色
    jiakuan = length/2
    for i in range(y):
        for j in range(x):
            pixNewImage[j+jiakuan,i+jiakuan] = (pixImg[j,i][0], pixImg[j,i][1], pixImg[j,i][2])
    return newImage


#缩放图片到指定大小
def suofangImage(img,length,width):
    imga = img.resize((length,width))
    return imga

#二值化
def erzhihua(img, rgb):
    #img = Image.open(filename)
    #img = img.convert("RGBA")
    img = img.convert("RGB")
    pixdata = img.load()

    for y in xrange(img.size[1]):
        for x in xrange(img.size[0]):
            if pixdata[x, y][0] + pixdata[x, y][1] + pixdata[x, y][2] > rgb*3:
                pixdata[x, y] = (0, 0, 0)
            else:
                pixdata[x, y] = (255, 255, 255)
    return img


#检测是否是白块
def isWhite(img):
    img = img.convert("RGB")
    pixdata = img.load()
    num = img.size[0] * img.size[1]
    count = 0
    for y in xrange(img.size[1]):
        for x in xrange(img.size[0]):
            if pixdata[x, y][0] + pixdata[x, y][1] + pixdata[x, y][2] > 200*3:
                count = count + 1

    if(num <= count+1):
        return True
    return False


def main(images, siteId, sessMap, grapMap):
    final_reco_text = []  # 存储最后识别出来的文字串
    #获取checkpoint的路径
    checkpoint_dir = FLAGS.checkpoint_Path+"checkpoint/checkpoint"+siteId+"/"
    label_dir = FLAGS.checkpoint_Path+"chineseLabels/chineseLabels"+siteId
    label_dict = get_label_dict(label_dir)
    #把要识别的图片标准化
    for image in images:
        if(not isWhite(image)):
            img = ImageChangeSize(image, 64, 64, 2)
            # imageList.append(img)
            final_predict_val, final_predict_index = inference(img, checkpoint_dir, sessMap, grapMap, siteId)
            # 给出top 3预测，candidate1是概率最高的预测
            for i in range(len(final_predict_val)):
                candidate1 = final_predict_index[i][0][0]
                #candidate2 = final_predict_index[i][0][1]
                #candidate3 = final_predict_index[i][0][2]
                final_reco_text.append(label_dict[int(candidate1)])
        else:
            final_reco_text.append(" ")
    return final_reco_text


def domain(images, siteId, sessMap, grapMap):
    logger.info('==================tensorFlow Start================')
    start = time.time();
    getParameter(siteId)
    text = main(images, siteId, sessMap, grapMap)
    result = ""
    for i in range(len(text)):
        result = result + text[i]
    logger.info(time.time() - start)
    logger.info('结果: '+result)
    logger.info('==================tensorFlow Finished================')
    return result


if __name__ == "__main__":
    # images = []
    # post_body = "{\"site\":\"40001\",\"images\":\"iVBORw0KGgoAAAANSUhEUgAAACYAAAAkCAYAAADl9UilAAAAvUlEQVR42u2YSw6AIAxEe/9LY1yQGBFsmRnACEk3RMujHwq1VBlmlljj1JXF/Q9b4X/Brso+Ddb6VgaGWmWDbbB7oHsksvASYKg0XRm1zPJgSLzSwSKLwlnZ60YpGNtaGyzPswp8GOxN+VAwT1lhlysIrDU/5IBl1EI5WE/2SVwZLdxT7vzLgrFfSnKwKTGmdKn0zo/AyYIfjTkJGGUR9u2iBqY48Qth9SskYL2Zx2zuPWzWqC0nGqD3CTZ6HODFGcgijBFfAAAAAElFTkSuQmCC  \"}"
    # data = json.loads(post_body)
    # print ("data json:{0}".format(data))
    # site = data["site"]
    # image_datas = data['images']
    #
    # str = image_datas
    # strArr = str.split()
    # for str in strArr:
    #     missing_padding = 4 - len(str) % 4
    #     if missing_padding:
    #         str += b'=' * missing_padding
    #     imageByte = base64.b64decode(str)
    #     img = Image.open(StringIO(imageByte))
    #     for i in range(1):
    #         images.append(img)
    # sessMap = {}
    # grapMap = {}
    # domain(images, site, sessMap, grapMap)


    sessMap = {}
    grapMap = {}
    siteId = "40001"
    filename = "/Users/shangzhen/PycharmProjects/CPS-OCR-Engine-master/ocr/tmp/43AAA5FAE3DFD76611F5105872A0E474.png"
    for i in range(2):
        time1 = time.time()
        img = Image.open(filename)
        getParameter(siteId)
        #传图片,返回的是list集合
        imgList = []
        imgList.append(img)

        result = domain(imgList, siteId, sessMap, grapMap)
        print(time.time() - time1)
        print result