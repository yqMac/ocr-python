from StringIO import StringIO

import numpy
from PIL import Image

from theano_ocr.training_data_gen.image_preprocessor import ImagePreprocessor


def read_and_parse(file_content, cracker, w, h):
    im = read_data(file_content, w, h)
    array = numpy.asarray(im.convert('L')).copy()
    image_input = ImagePreprocessor.ProcessImage(array)
    predicted_chars, char_probabilities = cracker.InferForImageArray(image_input)
    return "".join(x for x in predicted_chars if x != 'unk')


def read_data(file_content, w, h):
    im = Image.open(StringIO(file_content))
    im = im.resize((int(w), int(h)), Image.ANTIALIAS)
    return im
