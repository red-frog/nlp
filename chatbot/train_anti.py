#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import random
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm


"""抗元模型训练"""


def test(params):

    from sequence_to_sequence import SequenceToSequence
    from data_utils import batch_flow_bucket as batch_flow
    from word_sequence import WordSequence
    from thread_generator import ThreadGenerator

    x_data, y_data = pickle.load(open('chatbot.pkl', 'rb'))
    ws = pickle.load(open('ws.pkl', 'rb'))

    n_epoch = 2   # 几轮训练
    batch_size = 128
    steps = int(len(x_data) / batch_size) + 1   # 训练步数
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )

    save_path = './model/s2ss_chatbot_anti.ckpt'

    tf.reset_default_graph()

    with tf.Graph().as_default():
        random.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)      # 初始化一个图

        with tf.Session(config=config) as sess:
            model = SequenceToSequence(
                input_vocab_size=len(ws),
                target_vocab_size=len(ws),
                batch_size=batch_size,
                **params
            )

            init = tf.global_variables_initializer()
            sess.run(init)

            flow = ThreadGenerator(
                batch_flow([x_data, y_data], ws, batch_size, add_end=[False, True]),
                queue_maxsize=30
            )

            dummy_encoder_inputs = np.array([
                np.array([WordSequence.PAD]) for _ in range(batch_size)
            ])
            dummy_encoder_length = np.array([1] * batch_size)

            for epoch in range(1, n_epoch+1):
                costs = list()
                bar = tqdm(range(steps), total=steps,
                           desc='epoch {}, loss=0.000000'.format(epoch))
                for _ in bar:
                    x, xl, y, yl = next(flow)
                    # [1, 2], [3, 4]
                    # [3, 4], [1, 2]
                    x = np.flip(x, axis=1)  # 进行翻转, 第一个参数, 翻转什么数据, 第二个参数, 翻转几次

                    add_loss = model.train(sess,
                                           dummy_encoder_inputs,
                                           dummy_encoder_length,
                                           y, yl, loss_only=True)

                    add_loss *= 0.5

                    cost, lr = model.train(sess, x, xl, y, yl, return_lr=True, add_loss=add_loss)
                    costs.append(cost)
                    bar.set_description('epoch {} loss={:.6f} lr={:.6f}'.format(
                        epoch,
                        np.mean(costs),
                        lr
                    ))  # 保留6位小数

                model.save(sess=sess, save_path=save_path)
            flow.close()

    # 测试
    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=12,
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

    # 第二次预测
    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=1,
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


def mian():
    import json
    test(params=json.load(open('params.json')))
