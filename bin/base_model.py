# -*- coding: utf-8 -*-
"""
@Time ： 2022/3/15 10:15
@Auth ： CC
@File ：base_model.py
@IDE ：PyCharm
@Motto：Talk is cheap. Show me the code.

"""
import json
import numpy as np
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import *

set_gelu('tanh')  # 切换tanh版本
num_classes = 119
maxlen = 128
batch_size = 32
def build_bert(config_path,checkpoint_path,model_type,is_display=False):
    # 加载预训练模型
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model=model_type,
        return_keras_model=False,
    )

    output = Lambda(lambda x: x[:, 0])(bert.model.output)
    output = Dense(units=num_classes,
                   activation='softmax',
                   kernel_initializer=bert.initializer)(output)

    model = keras.models.Model(bert.model.input, output)

    if is_display:
        model.summary()
    return model
