#! /usr/bin/env python
# coding=utf-8
import logging
import time
from logging.handlers import TimedRotatingFileHandler

FOREGROUND_WHITE = 0x0007
FOREGROUND_BLUE = 0x01  # text color contains blue.
FOREGROUND_GREEN = 0x02  # text color contains green.
FOREGROUND_RED = 0x04  # text color contains red.
FOREGROUND_YELLOW = FOREGROUND_RED | FOREGROUND_GREEN

STD_OUTPUT_HANDLE = -11


# std_out_handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)


# def set_color(color, handle=std_out_handle):
# bool = ctypes.windll.kernel32.SetConsoleTextAttribute(handle, color)
# return bool


class Logger:
    def __init__(self, path, clevel=logging.DEBUG, Flevel=logging.DEBUG):
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        # 设置CMD日志
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(clevel)
        # 滚动日期切割
        log_file_handler = TimedRotatingFileHandler(filename=path, when="D", interval=1, backupCount=7)
        log_file_handler.suffix = "%Y-%m-%d_%H-%M.logs"
        # log_file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}.log$")
        log_file_handler.setFormatter(fmt)
        log_file_handler.setLevel(Flevel)

        self.logger.addHandler(sh)
        # self.logger.addHandler(fh)
        self.logger.addHandler(log_file_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def war(self, message, color=FOREGROUND_YELLOW):
        # set_color(color)
        self.logger.warn(message)
        # set_color(FOREGROUND_WHITE)

    def error(self, message, color=FOREGROUND_RED):
        # set_color(color)
        self.logger.error(message)
        # set_color(FOREGROUND_WHITE)

    def cri(self, message):
        self.logger.critical(message)


if __name__ == '__main__':

    logyyx = Logger('yyx.logs', logging.INFO, logging.INFO)
    # count = 0
    # while True:
        # count = count + 1
    timeStr = time.ctime()
    logyyx.debug('一个debug信息' + timeStr)
    logyyx.info('一个info信息' + timeStr)
    logyyx.war('一个warning信息' + timeStr)
    logyyx.error('一个error信息' + timeStr)
    logyyx.cri('一个致命critical信息' + timeStr)

    # time.sleep(5)
