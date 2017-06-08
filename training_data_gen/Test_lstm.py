import model
from model.captcha_cracker import CaptchaCracker
import theano
import os
import random

theano.config.floatX = "float64"
lstm_model_params_prefix = '/home/aixiang/Workspaces/image/lstm/lstm'
cracker = model.captcha_cracker.CaptchaCracker(lstm_model_params_prefix, includeCapital=False, num_rnn_steps=6)
image_dir = '/home/aixiang/Workspaces/image/ydImg'

files = os.listdir(image_dir)
for j in files:
    file_path = os.path.join(image_dir, j)
    print file_path
    predicted_chars, char_probabilities = cracker.InferFromImagePath(file_path)
    print "".join(x for x in predicted_chars)
    # for i in range(len(predicted_chars)):
    #     print 'predicted_char= {0}'.format(predicted_chars[i])
    #     print sorted([(char, prob) for char, prob in char_probabilities[i].iteritems()], key=lambda x: x[1],reverse=True)[:10]
