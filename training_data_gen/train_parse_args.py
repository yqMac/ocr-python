import argparse
import sys


def train_parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", help="Path where captcha is ")
    parser.add_argument("max_size",type=int, help="Num of captcha to save as one model")
    parser.add_argument("length",type=int, help="Length of the characters")
    parser.add_argument("-case_sensitive", help="if case sensitive",default=False)
    parser.add_argument("-captcha_pattern", help="capthcha filename pattern",default="^\d+_(.*)\..+$")
    args = parser.parse_args()
    print "Params passed are : "
    print args
    return args
