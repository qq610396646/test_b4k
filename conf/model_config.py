# -*- coding: utf-8 -*-
"""
@Time ： 2022/3/16 10:37
@Auth ： CC
@File ：model_config.py
@IDE ：PyCharm
@Motto：Talk is cheap. Show me the code.

"""
import os
dev_data_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/data/dev.json"
train_data_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/data/train.json"
model_base=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/model/"
onnx_model_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/onnx_model/"

ab_base_config={
    "config_path":model_base+"albert_base/albert_config.json",
    "checkpoint_path":model_base+"albert_base/model.ckpt-best",
    "dict_path":model_base+"albert_base/vocab_chinese.txt",
    "model_type":'albert',
    "is_display":False
}

ab_large_config={
    "config_path":model_base+"albert_large/albert_config.json",
    "checkpoint_path":model_base+"albert_large/model.ckpt-best",
    "dict_path":model_base+"albert_large/vocab_chinese.txt",
    "model_type":'albert',
    "is_display":False
}

ab_xlarge_config={
    "config_path":model_base+"albert_xlarge/albert_config.json",
    "checkpoint_path":model_base+"albert_xlarge/model.ckpt-best",
    "dict_path":model_base+"albert_xlarge/vocab_chinese.txt",
    "model_type":'albert',
    "is_display":False
}

bert_base_config={
    "config_path":model_base+"bert_base/bert_config.json",
    "checkpoint_path":model_base+"bert_base/bert_model.ckpt",
    "dict_path":model_base+"bert_base/vocab.txt",
    "model_type":'bert',
    "is_display":False
}

bert_wwm_config={
    "config_path":model_base+"bert_wwm/bert_config.json",
    "checkpoint_path":model_base+"bert_wwm/bert_model.ckpt",
    "dict_path":model_base+"bert_wwm/vocab.txt",
    "model_type":'bert',
    "is_display":False
}

bert_wwm_ext_config={
    "config_path":model_base+"bert_wwm_ext/bert_config.json",
    "checkpoint_path":model_base+"bert_wwm_ext/bert_model.ckpt",
    "dict_path":model_base+"bert_wwm_ext/vocab.txt",
    "model_type":'bert',
    "is_display":False
}

rb_plus_small_config={
    "config_path":model_base+"roberta_plus_small/bert_config.json",
    "checkpoint_path":model_base+"roberta_plus_small/bert_model.ckpt",
    "dict_path":model_base+"roberta_plus_small/vocab.txt",
    "model_type":'bert',
    "is_display":False
}

rb_plus_tiny_config={
    "config_path":model_base+"roberta_plus_tiny/bert_config.json",
    "checkpoint_path":model_base+"roberta_plus_tiny/bert_model.ckpt",
    "dict_path":model_base+"roberta_plus_tiny/vocab.txt",
    "model_type":'bert',
    "is_display":False
}

rb_small_config={
    "config_path":model_base+"roberta_small/bert_config.json",
    "checkpoint_path":model_base+"roberta_small/bert_model.ckpt",
    "dict_path":model_base+"roberta_small/vocab.txt",
    "model_type":'bert',
    "is_display":False
}

rb_tiny_config={
    "config_path":model_base+"roberta_tiny/bert_config.json",
    "checkpoint_path":model_base+"roberta_tiny/bert_model.ckpt",
    "dict_path":model_base+"roberta_tiny/vocab.txt",
    "model_type":'bert',
    "is_display":False
}

rb_wwm_ext_config={
    "config_path":model_base+"roberta_wwm_ext/bert_config.json",
    "checkpoint_path":model_base+"roberta_wwm_ext/bert_model.ckpt",
    "dict_path":model_base+"roberta_wwm_ext/vocab.txt",
    "model_type":'bert',
    "is_display":False
}

rb_wwm_large_ext_config={
    "config_path":model_base+"roberta_wwm_large_ext/bert_config.json",
    "checkpoint_path":model_base+"roberta_wwm_large_ext/bert_model.ckpt",
    "dict_path":model_base+"roberta_wwm_large_ext/vocab.txt",
    "model_type":'bert',
    "is_display":False
}


sb_base_config={
    "config_path":model_base+"simbert_base/bert_config.json",
    "checkpoint_path":model_base+"simbert_base/bert_model.ckpt",
    "dict_path":model_base+"simbert_base/vocab.txt",
    "model_type":'bert',
    "is_display":False
}

sb_small_config={
    "config_path":model_base+"simbert_small/bert_config.json",
    "checkpoint_path":model_base+"simbert_small/bert_model.ckpt",
    "dict_path":model_base+"simbert_small/vocab.txt",
    "model_type":'bert',
    "is_display":False
}

sb_tiny_config={
    "config_path":model_base+"simbert_tiny/bert_config.json",
    "checkpoint_path":model_base+"simbert_tiny/bert_model.ckpt",
    "dict_path":model_base+"simbert_tiny/vocab.txt",
    "model_type":'bert',
    "is_display":False
}

config_dict={
    #"albert_base":ab_base_config,
    #"albert_large":ab_large_config,
    # "albert_xlarge":ab_xlarge_config,
    # "bert_base":bert_base_config,
    # "bert_wwm":bert_wwm_config,
    # "bert_wwm_ext":bert_wwm_ext_config,
    # "roberta_plus_small":rb_plus_small_config,
    # "roberta_plus_tiny":rb_plus_tiny_config,
    # "roberta_small":rb_small_config,
    # "roberta_tiny":rb_tiny_config,
    # "roberta_wwm_ext":rb_wwm_ext_config,
    "roberta_wwm_large_ext":rb_wwm_large_ext_config,
    "simbert_base":sb_base_config,
    "simbert_small":sb_small_config,
    "simbert_tiny":sb_tiny_config
}
