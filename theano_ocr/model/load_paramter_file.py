import numpy as np


def load_file():
    filename = "/Users/rookie/antiSpider/imgsrc/ydzhejiang/result/training_images_9.npy.npz"
    filename_new = "/Users/rookie/antiSpider/imgsrc/ydzhejiang/result/training_images_9.npy.npz"
    file_path = "/Users/rookie/antiSpider/imgsrc/ydzhejiang/result/training_images_911.npy.npz"
    with np.load(filename) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        print len(f.files), len(param_values)
        del f.f
    with np.load(filename_new) as f:
        param_values_new = [f['arr_%d' % i] for i in range(len(f.files))]
        print len(f.files), len(param_values_new)
        del f.f
    x = 0
    while (x < len(param_values_new)):
        param_values_new[x] = param_values[x]
    x += 1

    np.savez(file_path, *param_values_new)

    with np.load(file_path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        print len(f.files), len(param_values)
        for x in param_values:
            print x.shape
        del f.f


load_file()
