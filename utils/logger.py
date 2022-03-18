# -*- coding: utf-8 -*-
"""
@Time ： 2022/3/17 13:23
@Auth ： CC
@File ：logger.py
@IDE ：PyCharm
@Motto：Talk is cheap. Show me the code.

"""
import logging


def logger_build(path="../log/log.txt"):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(' %(asctime)s - %(filename)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


logger = logger_build()

if __name__ == '__main__':
    print("test logger")

    logger = logger_build()
    logger.info("infologger")
    logger.warning("warningslogger")

