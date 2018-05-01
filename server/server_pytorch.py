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
# import inference
from rookie_utils import mod_config
from rookie_utils.Logger import Logger
import  logging

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
        state_dict = torch.load(model_path + one, lambda storage, loc: storage)
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            logger.error("model format error: {}, try parallel model".format(e.message))
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict)
            # model.load_state_dict(torch.load(model_path + one, lambda storage, loc: storage))
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
            print ('ignore lstm')
            # addTheanoModel(one)
            # t = threading.Thread(target=addTheanoModel, args=(one,))
            # t.start()
        elif one.startswith("crnn"):
            addCRNNModel(one)
            # t = threading.Thread(target=addCRNNModel, args=(one,))
            # t.start()
        else:
            print("格式无法匹配模型theano或者crnn: {}".format(one))
    # 启动文件夹监听服务
    # global observer
    # event_handler = FileEventHandler()
    # observer.schedule(event_handler, watcher_path, True)
    # observer.start()
    # observer.join()


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
                        # code = inference.read_and_parse(image_data, cracker, w, h)
                        # result["result"] = code
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
        # elif '/pyfont' == self.path:
        #     content_len = int(self.headers.getheader('content-length'))
        #     post_body = self.rfile.read(content_len)
        #     data = json.loads(post_body)
        #     logger.info("data json:{0}".format(data))
        #     if (not data.has_key("font")) or data['font'] == "":
        #         result["success"] = False
        #         result["msg"] = "font"
        #     elif not data.has_key("uuid"):
        #         result["msg"] = "uuid"
        #     else:
        #         try:
        #             site = data["uuid"]
        #             if '' == site:
        #                 site = uuid.uuid1()
        #             image_data = data['font']
        #             missing_padding = 4 - len(image_data) % 4
        #             if missing_padding:
        #                 image_data += b'=' * missing_padding
        #             image_data = base64.b64decode(image_data)
        #             f = open(font_path + site + ".woff", 'w')
        #
        #             f.write(image_data)
        #             f.close()
        #             font = TTFont(font_path + site + ".woff")
        #             font.saveXML(font_path + site + ".xml")
        #             fontXML = open(font_path + site + ".xml")
        #             fontXMLData = fontXML.read()
        #             fontXML.close()
        #             resData = base64.encodestring(fontXMLData)
        #             result["result"] = resData
        #             result["success"] = True
        #         except Exception:
        #             logger.info("pyfont 异常:{0}".format(Exception.message))
        #             result["msg"] = "识别过程发生异常"
        #             result["ex"] = Exception.message
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
# watcher_path = project_path + mod_config.getConfig("watcher_params", "path")
# logger.info('要监听的文件夹：{0}'.format(watcher_path))

list_model = os.listdir(model_path)
# 所有model的缓存
cracker_map = {}

# 监听服务
# observer = Observer()
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
        # observer.stop()
        sys.exit()
