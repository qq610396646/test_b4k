# -*- coding: utf-8 -*-
"""
@Time ： 2022/3/15 10:12
@Auth ： CC
@File ：origin_gpu.py
@IDE ：PyCharm
@Motto：Talk is cheap. Show me the code.

"""
from functools import wraps
from base_model import build_bert
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.backend import keras
from keras.utils import multi_gpu_model
from data_generator import data_generator,load_data
from time import perf_counter
import os
from conf.model_config import config_dict,dev_data_path,train_data_path
from utils.logger import logger

valid_data = load_data(dev_data_path)
train_data = load_data(train_data_path)
maxlen = 128
batch_size=16

def costtime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        s=perf_counter()
        for i in range(5):
            func(*args, **kwargs)
        e=perf_counter()
        logger.info(f"耗时：{e-s}")
        return
    return wrapper

@costtime
def model_predict(config,is_GPU=False,GPUS=1):
    model=build_bert(
        config_path=config["config_path"],
        checkpoint_path=config["checkpoint_path"],
        model_type=config["model_type"],
    )
    if is_GPU:
        model=multi_gpu_model(model,GPUS)
    tokenizer = Tokenizer(config["dict_path"], do_lower_case=True)
    valid_generator = data_generator(valid_data,batch_size)
    valid_generator.set_tokenizer(tokenizer,maxlen)

    count=0
    right=0
    for data,label in valid_generator:
        count +=1
        y_pred = model.predict(data).argmax(axis=1)
        right += (label == y_pred).sum()
    auc=right/count/batch_size*100
    print(f"准确率:{auc:.2f}%")
    del model

@costtime
def model_train(config,is_GPU=False,GPUS=1):
    tokenizer = Tokenizer(config["dict_path"], do_lower_case=True)
    train_generator = data_generator(train_data,batch_size)
    train_generator.set_tokenizer(tokenizer, maxlen)
    valid_generator = data_generator(valid_data, batch_size)
    valid_generator.set_tokenizer(tokenizer, maxlen)

    model=build_bert(
        config_path=config["config_path"],
        checkpoint_path=config["checkpoint_path"],
        model_type=config["model_type"],
    )

    if is_GPU:
        model=multi_gpu_model(model,GPUS)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(5e-5),
        metrics=['accuracy'],
    )
    model.fit_generator(train_generator.forfit(),
                        steps_per_epoch=len(train_generator),
                        epochs=10)


    count=0
    right=0
    for data,label in valid_generator:
        count +=1
        y_pred = model.predict(data).argmax(axis=1)
        right += (label == y_pred).sum()
    auc=right/count/batch_size*100
    print(f"准确率:{auc:.2f}%")
    del model

if __name__ == '__main__':


    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # roberta_tiny
    # runmodel(rb_tiny_config,False,4)
    for model_name,model_config in config_dict.items():
        logger.info(f"{model_name}开始测试")
        model_predict(model_config, False, 1)
    # model_train(config_dict["albert_base"],True,4)

