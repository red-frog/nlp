#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import datetime
import sys
import random
import numpy as np
import pickle
import tensorflow as tf
from tqdm import tqdm


def test(params):
    from sequence_to_sequence import SequenceToSequence
    from data_utils import batch_flow_bucket as batch_flow
    from thread_generator import ThreadGenerator

    x_data, y_data = pickle.load(open('chatbot.pkl', 'rb'))
    ws = pickle.load(open('ws.pkl', 'rb'))

    # 训练
    """
    1. n_epoch:训练轮次数
    2. 理论上训练轮次数越大, 那么训练精度越高
    3. 如果轮次特别大,比如1000, 那么可能发生过拟合, 但是是否过拟合也和训练数据有关
    4. n_epoch越大, 训练时间越长
    5. 用P5000的GPU训练40轮,训练3天,训练2轮, 大概一个半小时,
        如果使用CPU, 速度会特别慢, 可能一轮就要几个小时
    """
    n_epoch = 2
    batch_size = 128
    steps = int(len(x_data) / batch_size) + 1
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = './model/s2ss_chatbot.ckpt'

    tf.reset_default_graph()  # 重置默认的图
    with tf.Graph().as_default():
        random.seed(0),
        np.random.seed(0),
        tf.set_random_seed(0)
        print('{} start train model'.format(datetime.datetime.now()))
        with tf.Session(config=config) as sess:
            # 定义模型
            model = SequenceToSequence(
                input_vocab_size=len(ws),
                target_vocab_size=len(ws),
                batch_size=batch_size,
                **params
            )
            print('{} init model success'.format(datetime.datetime.now()))
            init = tf.global_variables_initializer()  # 初始化
            sess.run(init)
            print('{} start more thread to train'.format(datetime.datetime.now()))
            flow = ThreadGenerator(
                batch_flow([x_data, y_data], ws, batch_size, add_end=[False, True]),
                queue_maxsize=20
            )

            for epoch in range(1, n_epoch+1):
                costs = list()
                bar = tqdm(range(steps), total=steps,
                           desc='epoch {}, loss=0.000000'.format(epoch))
                for _ in bar:
                    x, xl, y, yl = next(flow)
                    # [1, 2], [3, 4]
                    # [3, 4], [1, 2]
                    x = np.flip(x, axis=1)  # 进行翻转, 第一个参数, 翻转什么数据, 第二个参数, 翻转几次
                    cost, lr = model.train(sess, x, xl, y, yl, return_lr=True)
                    costs.append(cost)
                    bar.set_description('epoch {} loss={:.6f} lr={:.6f}'.format(
                        epoch,
                        np.mean(costs),
                        lr
                    ))  # 保留6位小数

                model.save(sess=sess, save_path=save_path)

    # 测试
    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=12,
        parallel_iterations=1,
        **params
    )

    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)  # 从save_path reload

        bar = batch_flow([x_data, y_data], ws, 1, add_end=False)
        t = 0
        for x, xl, y, yl in bar:
            x = np.flip(x, axis=1)
            pred = model_pred.predict(
                sess,
                np.array(x),
                np.array(xl)
            )
            print(ws.inverse_transform(x[0]))
            print(ws.inverse_transform(y[0]))
            print(ws.inverse_transform(pred[0]))
            t += 1
            if t >= 3:
                break


def main():
    import json
    test(json.load(open('params.json')))


if __name__ == '__main__':
    main()

