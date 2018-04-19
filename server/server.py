# coding:utf-8

import base64
import json
import os
import sys
import threading
from BaseHTTPServer import BaseHTTPRequestHandler
from BaseHTTPServer import HTTPServer
from io import BytesIO
from SocketServer import ThreadingMixIn
import torch
from PIL import Image
from torch.autograd import Variable

sys.path.append("..")
import crnn_pytorch.dataset as dataset
import crnn_pytorch.models.crnn as crnn
import crnn_pytorch.utils as utils
import inference
import theano_ocr.model as theano_model
from rookie_utils import mod_config
from rookie_utils.Logger import Logger
from rookie_utils.models_watcher import *
from theano_ocr.model.captcha_cracker import CaptchaCracker
from fontTools.ttLib import TTFont
import uuid

alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
converter = utils.strLabelConverter(alphabet)
transformer = dataset.resizeNormalize((100, 32))


# 用于统计当前model的文件夹中有多少model文件
def list_model_count():
    lists = os.listdir(model_path)
    count = 0
    for one in lists:
        if not one.startswith(".") and one.endswith(".npz"):
            count += 1
        if not one.startswith(".") and one.endswith(".pth"):
            count += 1
    return count


# 监听文件夹的处理方法
class FileEventHandler(FileSystemEventHandler):
    def __init__(self):
        FileSystemEventHandler.__init__(self)

    def on_moved(self, event):
        if event.is_directory:
            logger.info(("directory moved from {0} to {1}".format(event.src_path, event.dest_path)))
        else:
            logger.info(("file moved from {0} to {1}".format(event.src_path, event.dest_path)))

    def on_created(self, event):
        if event.is_directory:
            logger.info(("directory created:{0}".format(event.src_path)))
        else:
            logger.info(("file created:{0}".format(event.src_path)))
            file_name = os.path.split(os.path.realpath(event.src_path))[1]
            statinfo = os.stat(event.src_path)
            msize = statinfo.st_size
            max_wait_loop = 0
            # 最多等待1个小时 360个10秒为1小时
            while max_wait_loop < (6 * 10 * 6):
                time.sleep(10)
                statinfo = os.stat(event.src_path)
                if statinfo.st_size - msize == 0:
                    logger.info('认为新model:{0}传输完成，最终大小{1},30秒后开始加载'.format(file_name, (statinfo.st_size)))
                    time.sleep(30)
                    if file_name.startswith("lstm"):
                        addTheanoModel(file_name)
                    elif file_name.startswith("crnn"):
                        addCRNNModel(file_name)
                    else:
                        logger.info("文件名称有误，请重新上传：{}".format(file_name))
                    break
                msize = statinfo.st_size
                max_wait_loop = max_wait_loop + 1

    def on_deleted(self, event):
        if event.is_directory:
            logger.info(("directory deleted:{0}".format(event.src_path)))
        else:
            logger.info(("file deleted:{0}".format(event.src_path)))

    def on_modified(self, event):
        if event.is_directory:
            logger.info(("directory modified:{0}".format(event.src_path)))
        else:

            logger.info(("file modified:{0}".format(event.src_path)))


# 加载Model
def addTheanoModel(one):
    start_time = time.clock()

    # global model_lock
    id = ''
    w = 80
    h = 150
    steps = 8
    version = "1.0"
    model_data = {}

    try:
        logger.info("start init cracker:{0}".format(one))
        if not one.endswith(".npz"):
            logger.error("{0}不以.npz结尾，不进行初始化".format(one))
            return

        if one.startswith("rookie"):
            vs = one.split("_")
            id = vs[1]
            w = 150
            h = 80
            steps = 8
            version = "2.0"
        else:
            h = one.split("_")[-5]
            w = one.split("_")[-4]
            id = one.split("_")[-3]
            steps = one.split("_")[-2]
            version = "1.0"
        logger.info("{0}分类完毕，version:{1},开始初始化".format(one, version))
        id_cracker = theano_model.captcha_cracker.CaptchaCracker(
            model_path + one,
            (None, 1, int(h), int(w)),
            includeCapital=False,
            multi_chars=True,
            rescale_in_preprocessing=False,
            num_rnn_steps=int(steps),
            use_mask_input=True)
        logger.info("{0}分类完毕，version:{1},初始化完毕".format(one, version))
        model_data = {
            "id": id,
            "w": w,
            "h": h,
            "steps": steps,
            "id_cracker": id_cracker,
            "file_name": one,
            "version": version,
            "type": "theano_ocr"
        }
        logger.info("{0}分类完毕，开始进行加载".format(one))
    except Exception, e:
        logger.error("{0}分类完毕，加载异常了:{1}".format(one, e.message))

    try:
        # 检测key是否已经存在
        if cracker_map.has_key(id):
            logger.info("key {0} 已经存在，文件 {1} 加载失败,解析后数据: {2}".format(id, one, model_data))
        else:
            cracker_map[id] = model_data
    except Exception, e:
        logger.error("{0}加载发生异常：{1}".format(one, e.message))
    finally:
        logger.info("{0}加载结束，释放锁".format(one))
    logger.info("finish init cracker:{0},spend{1}".format(id, time.clock() - start_time))


def addCRNNModel(one):
    try:
        logger.info('loading pretrained model from {}'.format(one))
        if not one.endswith(".pth"):
            return
        h = one.split("_")[-5]
        w = one.split("_")[-4]
        id = one.split("_")[-3]
        steps = one.split("_")[-2]
        version = "1.0"
        model = crnn.CRNN(32, 1, 37, 256)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path + one, lambda storage, loc: storage))
        model.eval()
        model_data = {
            "id": id,
            "w": w,
            "h": h,
            "steps": steps,
            "id_cracker": model,
            "file_name": one,
            "version": version,
            "type": "crnn"
        }
        cracker_map[id] = model_data
        logger.info('finished loading pretrained model {}'.format(one))
    except Exception, e:
        logger.info("加载异常: {}".format(e.message))


# 多线程加载，加快加载速度
def initModes():
    for one in list_model:
        if one.startswith("lstm"):
            addTheanoModel(one)
            # t = threading.Thread(target=addTheanoModel, args=(one,))
            # t.start()
        elif one.startswith("crnn"):
            addCRNNModel(one)
            # t = threading.Thread(target=addCRNNModel, args=(one,))
            # t.start()
        else:
            print("格式无法匹配模型theano或者crnn: {}".format(one))
    # 启动文件夹监听服务
    global observer
    event_handler = FileEventHandler()
    observer.schedule(event_handler, watcher_path, True)
    observer.start()
    observer.join()


# 处理网络请求
class GetHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "json")
        self.send_header("Content-Encoding", "utf-8")
        self.end_headers()
        logger.info(self.path)
        result = {"success": False}
        try:
            if '/pyocr/list' == self.path:
                list = []
                for key in cracker_map:
                    j = cracker_map[key]
                    data = {
                        "id": j["id"],
                        "图片宽度": j["w"],
                        "图片高度": j["h"],
                        "最大长度": j["steps"],
                        "模型文件": j["file_name"],
                        "模型版本": j["version"]
                    }
                    list.append(data)
                result['list'] = list
                result['count'] = len(list)
                result['total'] = list_model_count()
                result["success"] = True
            else:
                result["msg"] = "不支持的请求地址"
        except Exception, e:
            result['success'] = False
            result['ex'] = e.message
            result["msg"] = "识别过程发生异常"
        logger.info(json.dumps(result))
        self.wfile.write(json.dumps(result).encode("utf-8"))
        return

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "json")
        self.send_header("Content-Encoding", "utf-8")
        self.end_headers()
        result = {"success": False}
        # 验证码识别
        if '/pyocr' == self.path:
            content_len = int(self.headers.getheader('content-length'))
            post_body = self.rfile.read(content_len)
            data = json.loads(post_body)
            logger.info("data json:{0}".format(data))
            if (not data.has_key("image")) or data['image'] == "":
                result["success"] = False
                result["msg"] = "image不能为空"
            elif data.has_key("site") and not cracker_map.has_key(data["site"]):
                result["msg"] = "无匹配Site"
            elif not data.has_key("site"):
                result["msg"] = "site不能为空"
            else:
                try:
                    site = data["site"]
                    cracker_data = cracker_map[site]
                    image_data = data['image']
                    missing_padding = 4 - len(image_data) % 4
                    if missing_padding:
                        image_data += b'=' * missing_padding
                    image_data = base64.b64decode(image_data)
                    # model_data = {"id": id, "w": w, "h": h, "steps": steps, "id_cracker": id_cracker}
                    type = cracker_data["type"]
                    cracker = cracker_data["id_cracker"]
                    if type == "theano_ocr":
                        w = cracker_data["w"]
                        h = cracker_data["h"]
                        code = inference.read_and_parse(image_data, cracker, w, h)
                        result["result"] = code
                    else:
                        image = Image.open(BytesIO(image_data)).convert('L')
                        image = transformer(image)
                        if torch.cuda.is_available():
                            image = image.cuda()
                        image = image.view(1, *image.size())
                        image = Variable(image)
                        preds = cracker(image)

                        _, preds = preds.max(2)
                        preds = preds.transpose(1, 0).contiguous().view(-1)
                        preds_size = Variable(torch.IntTensor([preds.size(0)]))
                        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
                        result["result"] = sim_pred
                    result["success"] = True
                except Exception, e:
                    result["msg"] = "识别过程发生异常"
                    result["ex"] = e.message
        elif '/pyfont' == self.path:
            content_len = int(self.headers.getheader('content-length'))
            post_body = self.rfile.read(content_len)
            data = json.loads(post_body)
            logger.info("data json:{0}".format(data))
            if (not data.has_key("font")) or data['font'] == "":
                result["success"] = False
                result["msg"] = "font"
            elif not data.has_key("uuid"):
                result["msg"] = "uuid"
            else:
                try:
                    site = data["uuid"]
                    if '' == site:
                        site = uuid.uuid1()
                    image_data = data['font']
                    missing_padding = 4 - len(image_data) % 4
                    if missing_padding:
                        image_data += b'=' * missing_padding
                    image_data = base64.b64decode(image_data)
                    f = open(font_path + site + ".woff", 'w')

                    f.write(image_data)
                    f.close()
                    font = TTFont(font_path + site + ".woff")
                    font.saveXML(font_path + site + ".xml")
                    fontXML = open(font_path + site + ".xml")
                    fontXMLData = fontXML.read()
                    fontXML.close()
                    resData = base64.encodestring(fontXMLData)
                    result["result"] = resData
                    result["success"] = True
                except Exception:
                    logger.info("pyfont 异常:{0}".format(Exception.message))
                    result["msg"] = "识别过程发生异常"
                    result["ex"] = Exception.message
        else:
            result["msg"] = "不支持的请求地址"
        json_string = json.dumps(result)
        self.wfile.write(json_string.encode("utf-8"))
        return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass


# 设置上层【项目根目录】为配置文件 所在目录
mod_config.setPath(".")
# 项目目录
project_path = mod_config.getConfig("project", "path")
# 日志输出
log_path = project_path + mod_config.getConfig("logger", "file")
logger = Logger(log_path, logging.INFO, logging.INFO)
# 线程锁，防止防止model时出问题
# model_lock = thread.allocate_lock()
# 读取models所在目录配置
model_path = project_path + mod_config.getConfig("model_params", "path")

font_path = project_path + mod_config.getConfig("font_params", "path")
# 监控文件夹
watcher_path = project_path + mod_config.getConfig("watcher_params", "path")
logger.info('要监听的文件夹：{0}'.format(watcher_path))

list_model = os.listdir(model_path)
# 所有model的缓存
cracker_map = {}

# 监听服务
observer = Observer()
# 启线程加载Model
threading.Thread(target=initModes).start()

if __name__ == '__main__':
    # 读取服务发布端口
    port = mod_config.getConfig("server", "port")
    try:
        server = ThreadedHTTPServer(('0.0.0.0', int(port)), GetHandler)
        # server = HTTPServer(('0.0.0.0', int(port)), GetHandler)
        logger.info('Starting server with port:{0}'.format(port))
        server.serve_forever()
    except Exception, e:
        print('ex')
        print(e)
    finally:
        print('final')
        observer.stop()
        sys.exit()
