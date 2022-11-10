from flask import Flask, request
from threading import Thread
import json

from pcl_pangu.context import set_context
from pcl_pangu.model import alpha

from gevent import monkey
from gevent.pywsgi import WSGIServer
monkey.patch_all(thread=False)
from multiprocessing import cpu_count, Process

APP = Flask(__name__)


class MyThread(Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result   # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None


set_context(backend='mindspore')
config = alpha.model_config_npu(model='2B6')
model, config_model = alpha.load_model(config, input='四川的省会是?')


def process_result(result):
    if result.isspace():
        return "我不能理解，换个方式提问呢~"
    if result.startswith("上联:") and len(result.split('\n') > 1):
        return result.split('\n')[1:]
    return result


@APP.route('/health', methods=['GET'])
def health_func():
    return json.dumps({'health': 'true'}, indent=4)


@APP.route('/infer/text', methods=['POST'])
def inference_text():
    input = request.json
    question = input['question']
    result = alpha.inference_with_model(config, model, config_model, input=question)
    result = process_result(result)
    res_data = {
        "result": result
    }
    return json.dumps(res_data, indent=4)


def run(MULTI_PROCESS):
    if MULTI_PROCESS == False:
        WSGIServer(('0.0.0.0', 8080), APP).serve_forever()
    else:
        mulserver = WSGIServer(('0.0.0.0', 8080), APP)
        mulserver.start()

        def server_forever():
            mulserver.start_accepting()
            mulserver._stop_event.wait()

        # for i in range(cpu_count()):
        for i in range(1):
            p = Process(target=server_forever)
            p.start()


if __name__ == "__main__":
    run(True)
