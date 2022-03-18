# -*- coding: utf-8 -*-
"""
@Time ： 2022/3/18 10:13
@Auth ： CC
@File ：kerasonnxmodel.py
@IDE ：PyCharm
@Motto：Talk is cheap. Show me the code.

"""
import keras2onnx
import numpy as np
import tf2onnx
import onnxruntime as rt
from conf.model_config import config_dict, dev_data_path, onnx_model_path, rb_plus_tiny_config, sb_small_config
from bin.data_generator import data_generator,load_data
from bin.base_model import build_bert
from bert4keras.tokenizers import Tokenizer
import tensorflow as tf
from keras.models import load_model

valid_data = load_data(dev_data_path)
maxlen = 128
batch_size=1

def model_save(config, model_save_path):
    model=build_bert(
        config_path=config["config_path"],
        checkpoint_path=config["checkpoint_path"],
        model_type=config["model_type"],
        is_display=False
    )

    # onnx_model = keras2onnx.convert_keras(model, model.name)
    # content = onnx_model.SerializeToString()
    # sess = rt.InferenceSession(content)
    # print(sess.get_inputs())

    tokenizer = Tokenizer(config["dict_path"], do_lower_case=False)
    valid_generator = data_generator(valid_data,batch_size)
    valid_generator.set_tokenizer(tokenizer,maxlen)
    spec = (tf.TensorSpec((None, None), tf.float32,name="input0"),tf.TensorSpec((None, None), tf.float32,name="input1"))
    output_path = model.name + ".onnx"
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
    output_names = [n.name for n in model_proto.graph.output]
    print(output_names)
    input = [n.name for n in model_proto.graph.input]
    print(input)
    providers = ['CPUExecutionProvider']
    m = rt.InferenceSession(output_path)
    m.set_providers(['CUDAExecutionProvider'], [ {'device_id': 1}])

    for data, label in valid_generator:

        onnx_pred = m.run(None, {input[0]: np.array(data[0],dtype="float32"),
                                 input[1]: np.array(data[1],dtype="float32")})
        print(onnx_pred[0].argmax(1))
    # #     y_pred = model.predict(data)
        break


    # for data, label in valid_generator:



if __name__ == '__main__':
    model_save(sb_small_config,onnx_model_path)