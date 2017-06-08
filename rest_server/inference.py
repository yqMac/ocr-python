from StringIO import StringIO

import numpy
from PIL import Image

from training_data_gen.image_preprocessor import ImagePreprocessor
# theano.config.floatX = "float64"

def read_and_parse(file_content,cracker):
    im = read_data(file_content)
    array = numpy.asarray(im.convert('L')).copy() 
    image_input = ImagePreprocessor.ProcessImage(array)
    #print "saving image"
    # im.save("geetssssssss.png")
    predicted_chars, char_probabilities = cracker.InferForImageArray(image_input)
    return "".join(x for x in predicted_chars)
    # return "save ok ";
def read_data(file_content):
    im = Image.open(StringIO(file_content))
    return im
