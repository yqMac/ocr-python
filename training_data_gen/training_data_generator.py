# coding:utf-8

import numpy
import os
import re
import uuid
import sys
import train_parse_args
sys.path.append("..")
from training_data_gen.image_preprocessor import ImagePreprocessor
import training_data_gen.vocabulary as vocabulary
import training_data_gen.utils as utils


# CAPTCHA_FILENAME_PATTERN = re.compile('^\d+_(.*)\..+$')
# image_dir = "/home/aixiang/Workspaces/image/image"
# npz_dir = "/home/aixiang/Workspaces/image/train"
# image_dir = "/Users/rookie/antiSpider/imgsrc/ydzhejiang/success"
# npz_dir = "/Users/rookie/antiSpider/imgsrc/ydzhejiang/npzs"

# image_dir = "/Users/rookie/antiSpider/imgsrc/ydold/test"
# npz_dir = "/Users/rookie/antiSpider/imgsrc/ydold/npzs"
# image_dir = "/home/aixiang/Workspaces/image/train_img"
# npz_dir = "/home/aixiang/Workspaces/image/train"
# image_dir = "/home/aixiang/Workspaces/image/valid_img"
# npz_dir = "/home/aixiang/Workspaces/image/valid"
# image_dir = "/home/aixiang/Workspaces/image/test_img"
# npz_dir = "/home/aixiang/Workspaces/image/test"




def _ParseCaptchaFromImageFilename(image_filepath, captcha_pattern):
    image_filename = os.path.basename(image_filepath)
    match = captcha_pattern.match(image_filename)
    assert match is not None, image_filename
    return match.group(1)


def _GetCaptchaIdsFromImageFilename(image_filepath, captcha_pattern, case_sensitive):
    captcha_str = _ParseCaptchaFromImageFilename(image_filepath, captcha_pattern)
    captcha_ids = numpy.zeros(len(captcha_str), dtype=numpy.int32)
    for i, captcha_char in enumerate(captcha_str):
        CHAR_VOCABULARY, CHARS = vocabulary.GetCharacterVocabulary(case_sensitive)
        captcha_ids[i] = CHAR_VOCABULARY[captcha_char]
    return captcha_ids


def _GetShapeOfImagesUnderDir(captchas_dir):
    for captcha_filepath in utils.GetFilePathsUnderDir(captchas_dir):
        # if not captcha_filepath.endswith(".png"):
        #     continue
        image_data = ImagePreprocessor.GetImageData(captcha_filepath)
        return image_data.shape
    return None


class TrainingData(object):
    @classmethod
    def Save(cls, file_path, image_data, chars):
        print file_path
        numpy.savez(file_path, image_data=image_data, chars=chars)

    @classmethod
    def Load(cls, file_path, rescale_in_preprocessing=False):
        training_data = numpy.load(file_path)
        image_input = training_data['image_data']
        if rescale_in_preprocessing:
            for row in range(image_input.shape[0]):
                image_input[row, 0, :, :] = ImagePreprocessor.RescaleImageInput(
                    image_input[row, 0, :, :])
        else:
            image_input = ImagePreprocessor.NormalizeImageInput(image_input)
        ret = (image_input, training_data['chars'])
        del training_data.f
        training_data.close()
        return ret

    @classmethod
    def GenerateTrainingData(cls,
                             captchas_dir,
                             training_data_dir,
                             captcha_pattern,
                             max_size=3000,
                             max_captcha_length=5,
                             case_sensitive=False
                             ):
        os.makedirs(training_data_dir + "/allnpz")

        image_shape = _GetShapeOfImagesUnderDir(captchas_dir)
        training_data_shape = tuple(
            [max_size] + list(ImagePreprocessor.GetProcessedImageShape(image_shape)))
        training_image_data = numpy.zeros(training_data_shape, dtype=numpy.float32)
        training_labels = numpy.zeros((max_size, max_captcha_length),
                                      dtype=numpy.int32)
        i = 0
        for captcha_filepath in utils.GetFilePathsUnderDir(captchas_dir):
            try:
                image_data = ImagePreprocessor.GetImageData(captcha_filepath)
            except Exception as e:
                print e, captcha_filepath
                continue

            i += 1
            index = i % max_size
            training_image_data[index] = ImagePreprocessor.ProcessImage(image_data)
            captcha_ids = _GetCaptchaIdsFromImageFilename(captcha_filepath, captcha_pattern, case_sensitive)
            training_labels[index, :] = numpy.zeros(max_captcha_length,
                                                    dtype=numpy.int32)
            training_labels[index, :captcha_ids.shape[0]] = captcha_ids

            if i != 0 and (i % max_size) == 0:
                print 'Generated {0} examples.'.format(i)

            if i != 0 and i % max_size == 0:
                file_path = os.path.join(
                    training_data_dir + "/allnpz", "training_images_{0}.npy".format(i / max_size))
                try:
                    cls.Save(file_path, training_image_data, training_labels)
                except Exception as e:
                    print e
        formatNpzDir(training_data_dir)


def copyFile(src_dir, tar_dir, file_name,tar_name=""):
    sourceFile = os.path.join(src_dir, file_name)
    targetFile = os.path.join(tar_dir, file_name)
    if tar_name:
        targetFile = os.path.join(tar_dir, tar_name)
    if os.path.isfile(sourceFile):
        if not os.path.exists(tar_dir):
            os.makedirs(tar_dir)
    if not os.path.exists(targetFile) or (
                os.path.exists(targetFile) and (os.path.getsize(targetFile) != os.path.getsize(sourceFile))):
        open(targetFile, "wb").write(open(sourceFile, "rb").read())


def formatNpzDir(npz_path):
    npz_path_list = os.listdir(npz_path + "/allnpz")
    npz_list = []
    for one in npz_path_list:
        if one.endswith(".npz"):
            npz_list.append(one)
    all_count = len(npz_list)
    # train_count = int(all_count / 10.0 * 8)
    # test_count = int(all_count / 10.0 * 1)
    # validate_count = all_count - train_count - test_count
    # 只留两个，1个验证，1个测试
    train_count = all_count - 2
    test_count = 1
    validate_count = all_count - train_count - test_count

    if test_count == 0:
        train_count = train_count - 1
        test_count = 1

    print "共生成{0}个训练集，其中{1}个用于训练，{2}个用于验证，{3}用于测试".format(all_count, train_count, validate_count, test_count)
    src_dir = npz_path + "/allnpz"
    train_dir = npz_path + "/train_npz"
    validate_dir = npz_path + "/validate_npz"
    test_dir = npz_path + "/test_npz"
    for i, file_name in enumerate(npz_list):
        # if i < train_count:
        #     copyFile(src_dir, train_dir, file_name)
        # elif i < train_count + validate_count:
        #     copyFile(src_dir, validate_dir, file_name)
        # else:
        #     copyFile(src_dir, test_dir, file_name)
        if i < 1:
            copyFile(src_dir, validate_dir, file_name,"validate_npy.npz")
        elif i < 2:
            copyFile(src_dir, test_dir, file_name,"test_npy.npz")
        else:
            copyFile(src_dir, train_dir, file_name)
def main(args):
    # 上层目录
    pre_cwd = os.path.dirname(os.getcwd())
    dir_id = pre_cwd + "/" + str(uuid.uuid1()).replace("-", "")
    print dir_id

    image_dir = args.image_dir
    npz_dir = dir_id

    print "图片路径：{0}，训练集生成目录：{1}".format(image_dir, npz_dir)
    captcha_pattern = re.compile(args.captcha_pattern)

    TrainingData.GenerateTrainingData(image_dir, npz_dir, captcha_pattern, args.max_size, args.length, args.case_sensitive)
    # TrainingData.GenerateTrainingData(image_dir, npz_dir, captcha_pattern, 1000, args.length, args.ignore_case)

    # formatNpzDir("/Users/rookie/PyWorker/testtheano/a3ff4dcc4e6d11e7a7654c32758b0f1b")


if __name__ == '__main__':
    arguments = train_parse_args.train_parse_arg()
    main(arguments)
