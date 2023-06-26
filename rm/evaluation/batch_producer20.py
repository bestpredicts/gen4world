# -*- coding:utf-8 -*-
"""
读取prompt输入文件，请求张雷团队维护的外部大模型任务管理接口(kafka)
1. 同一批数据，因为优先级反复调整，需要多次重跑提交
2. 提交前首先需要张小佳配合，将队列中堆积的任务全部reset
3. 多个任务可以同时提交

每次计划获取数据时手动执行:
1. 设置此次任务的输入文件位置到 promptFilelist
2. 设置已经产出的输出文件位置到 hasDoneFilelist
3. 设置此次任务的全量结果文件位置到 newOutFile
4. 手动执行 # /nfs/v100-022/niuqiang002/bellaspace/tools/bin/python3 /nfs/v100-022/niuqiang002/bellaspace/code10/batch_producer20.py
5. 观察三者的数量是否符合预期。（比如队列长度小于输入文件行数），如果不符合，重跑
"""
import copy
import hashlib
import json
import logging
import os
import uuid
import time
from logging.handlers import TimedRotatingFileHandler
from kafka import KafkaProducer

logger = logging.getLogger("interlogger")
logger.setLevel(logging.INFO)
file_handler = TimedRotatingFileHandler(
    "/nfs/v100-022/niuqiang002/bellaspace/log10/producer20.log", "H", 24 * 14, encoding="utf-8")
file_handler.suffix = "%Y%m%d%H"
formater = logging.Formatter(
    '%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
file_handler.setFormatter(formater)
logger.addHandler(file_handler)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formater)
logger.addHandler(console)

# ------------------------------------------------------------------------------------------------
batchId = 'llama65b_4k_15epoch.submit.json'
promptFilelist = [
    f'/data/nfs14/nfs/aisearch/asr/dengyong013/project/nlp2agi/infer/data/{batchId}'
]
resultFilelist = [
    f'/nfs/v100-022/niuqiang002/bellaspace/data10/{batchId}.done']
temperature = 0.8
qpslimit = 3
# ------------------------------------------------------------------------------------------------
requestTmpl = {
    'channel': 10,
    'businessCode': "belle_batch",
    'batchId': "",
    'convId': "",
    'questionId': "",
    'prompt': "",
    'configProperties': {
        'temperature': temperature
    },
    "enableRetry": True,
    "retryTimes": 10
}


g_prompt_collection = dict()


def load_manual_prompt():
    """
    一次性读取，避免上游覆写数据
    """
    for promptfile in promptFilelist:
        if not os.path.isfile(promptfile):
            logger.info("unexist file={}".format(promptfile))
            continue
        with open(promptfile, "r") as promptfd:
            lineno = 0
            while True:
                try:
                    line = promptfd.readline()
                    if not line:
                        logger.info("eof of input={}".format(promptfile))
                        break
                    promptObj = json.loads(line)
                    requestObj = copy.deepcopy(requestTmpl)
                    # consumer会根据batchId写文件
                    requestObj["batchId"] = batchId
                    convid = str(promptObj["id"])
                    requestObj["convId"] = convid
                    requestObj["questionId"] = convid + \
                        "-" + str(uuid.uuid4())[:6]  # 避免张雷团队消息重复
                    requestObj["prompt"] = promptObj["prompt"]
                    g_prompt_collection[convid] = requestObj
                except Exception as e:
                    lineno += 1
                    logger.error("file={} lineno={} line={} exception={}".format(
                        promptfile, lineno, line, repr(e)))
                    continue
    logger.info("input length={}".format(len(g_prompt_collection)))


def load_structured_prompt():
    """
    一次性读取，避免上游覆写数据
    """
    for promptfile in promptFilelist:
        if not os.path.isfile(promptfile):
            logger.info("unexist file={}".format(promptfile))
            continue
        with open(promptfile, "r") as promptfd:
            lineno = 0
            while True:
                try:
                    line = promptfd.readline()
                    if not line:
                        logger.info("eof of input={}".format(promptfile))
                        break
                    promptObj = json.loads(line)
                    requestObj = copy.deepcopy(requestTmpl)
                    requestObj["businessCode"] = "belle_batch"
                    # consumer会根据batchId写文件
                    requestObj["batchId"] = batchId
                    convid = promptObj["convId"]
                    requestObj["convId"] = convid
                    requestObj["questionId"] = promptObj["questionId"] + \
                        "--" + str(uuid.uuid4())[:8]  # 避免张雷团队消息重复
                    requestObj["prompt"] = promptObj["prompt"]
                    configObj = promptObj["configProperties"]
                    if "max_tokens" in configObj:
                        popvalue = configObj.pop("max_tokens")
                    requestObj["configProperties"] = configObj
                    requestObj["enableRetry"] = True
                    requestObj["retryTimes"] = 5
                    g_prompt_collection[convid] = requestObj
                except Exception as e:
                    logger.error("file={} lineno={} line={} exception={}".format(
                        promptfile, lineno, line, repr(e)))
                    continue
    logger.info("input length={}".format(len(g_prompt_collection)))


def remove_alreadyDone():
    """
    剔除已经产出的结果
    """

    for resultfile in resultFilelist:
        if not os.path.isfile(resultfile):
            logger.info("unexist file={}".format(resultfile))
            continue
        with open(resultfile, "r") as resultfd:
            while True:
                line = resultfd.readline()
                if not line:
                    logger.info("### eof of result={}".format(resultfile))
                    break
                try:
                    resultObj = json.loads(line)
                    convid = resultObj["data"]["convId"]
                    if convid in g_prompt_collection.keys():
                        poped = g_prompt_collection.pop(convid)  # DEBUG
                    else:
                        pass
                        # logger.warning("cannot find in prompt convid={}".format(convid))
                except Exception as e:
                    logger.error("line={} exception={}".format(line, repr(e)))
                    continue
    logger.info("after remove done input length={}".format(
        len(g_prompt_collection)))


def check_and_send():
    """
    发送请求到队列
    """
    for convid, requestObj in g_prompt_collection.items():
        try:
            requestStr = json.dumps(requestObj, ensure_ascii=False)
            pong = producer.send(topic, requestStr.encode('utf-8'))

            time.sleep(1.0 / qpslimit)
        except Exception as e:
            logger.error("convid={} exception={}".format(convid, repr(e)))
            continue


topic = 'chatgpt-channel10-request'
broker = ['kafka118-online.zeus.ljnode.com:9092',
          'kafka119-online.zeus.ljnode.com:9092',
          'kafka120-online.zeus.ljnode.com:9092']
producer = KafkaProducer(bootstrap_servers=broker)

if __name__ == "__main__":
    logger.info("restarting. cation with args!!!")
    load_manual_prompt()
    try:
        remove_alreadyDone()
    except Exception as e:
        logger.info("reading donefile exception={}".format(repr(e)))
    check_and_send()
    logger.info("stopping")
