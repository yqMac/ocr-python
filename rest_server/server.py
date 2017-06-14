# coding:utf-8

import base64
import json
from BaseHTTPServer import BaseHTTPRequestHandler
import sys
sys.path.append("..")
import inference
import model
import os
from model.captcha_cracker import CaptchaCracker

# theano.config.floatX = "float64"

lstm_model_params_prefix = ("./lstm/lstm_")
list_model = os.listdir("./lstm/")
cracker_map = {}

# 加载所有训练好的模板
for one in list_model:
    if not one.endswith(".npz"):
        continue
    h = one.split("_")[-5]
    w = one.split("_")[-4]
    id = one.split("_")[-3]
    num = one.split("_")[-2]
    id_cracker = model.captcha_cracker.CaptchaCracker(
        os.getcwd()+"/lstm/" + one,
        (None, 1,int(h), int(w)),
        includeCapital=False,
        multi_chars=True,
        rescale_in_preprocessing=False,
        num_rnn_steps=int(num),
        use_mask_input=True)
    cracker_map[id] = id_cracker
    print "finish init cracker:{0}".format(id)


# cracker = model.captcha_cracker.CaptchaCracker(
#     lstm_model_params_prefix,
#     includeCapital=False,
#     multi_chars=True,
#     rescale_in_preprocessing=False,
#     num_rnn_steps=5,
#     use_mask_input=True)


class GetHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "json")
        self.send_header("Content-Encoding", "utf-8")
        self.end_headers()
        result = {}
        result["success"] = True

        content_len = int(self.headers.getheader('content-length'))
        # print "content length:{0}".format(content_len)
        post_body = self.rfile.read(content_len)

        data = json.loads(post_body)
        print "data json:{0}".format(data)
        if (not data.has_key("image"))or data['image']=="":
            result["success"] = False
            result["msg"] = "image不能为空"
            json_string = json.dumps(result)
            self.wfile.write(json_string)
            return
        # fix blank
        if data.has_key("site") and not cracker_map.has_key(data["site"]):
            result["success"] = False
            result["msg"] = "无匹配Site"
            json_string = json.dumps(result)
            self.wfile.write(json_string)
            return
        if not data.has_key("site"):
            result["success"] = False
            result["msg"] = "site不能为空"
            json_string = json.dumps(result)
            self.wfile.write(json_string)
            return
        site = data["site"]
        cracker = cracker_map[site]
        imageData = data['image']
        missing_padding = 4 - len(imageData) % 4
        if missing_padding:
            imageData += b'=' * missing_padding
        imageData = base64.b64decode(imageData)
        code  = inference.read_and_parse(imageData, cracker)
        result["result"]= code
        json_string = json.dumps(result)
        self.wfile.write(json_string)

        return


if __name__ == '__main__':
    from BaseHTTPServer import HTTPServer

    server = HTTPServer(('0.0.0.0', 8088), GetHandler)
    print 'Starting server, use <Ctrl-C> to stop'
    server.serve_forever()
