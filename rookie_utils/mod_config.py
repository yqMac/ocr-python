# encoding:utf-8
# name:mod_config.py

import ConfigParser
import os

global config_path
config_path = os.path.split(os.path.realpath(__file__))[0] + "/config"


def setPath(path):
    global config_path
    config_path = path + "/config"


# 获取config配置文件
def getConfig(section, key):
    config = ConfigParser.ConfigParser()
    # path = os.path.split(os.path.realpath(__file__))[0] + '/db.conf'
    config.read(config_path)
    return config.get(section, key)

# 其中 os.path.split(os.path.realpath(__file__))[0] 得到的是当前文件模块的目录
