#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
sys.setrecursionlimit(4000)
import numpy as np
import tensorflow as tf
from tensorflow import layers   # 构建层等

from tensorflow.python.ops import array_ops   # 数组操作
from tensorflow.contrib import seq2seq
from tensorflow.contrib.seq2seq import BahdanauAttention, LuongAttention, AttentionWrapper, BeamSearchDecoder

# RNN, LSTM

from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell, DropoutWrapper, ResidualWrapper

from word_sequence import WordSequence
from data_utils import _get_embed_device


class SequenceToSequence(object):
    """
    基本流程
    __init__:基本参数的保存, 参数验证(验证参数的合法性)
    build_model: 构建模型
    init_placeholders 初始化一些TensorFlow的变量的占位符
    build_encoder 初始化编码器
        build_single_cell
        build_decoder_cell
    init_optimizer 如果是在训练模式下进行, 则需要初始化优化器
    train 训练一个batch数据
    predict 预测一个batch数据
    """

    def __init__(self,
                 input_vocab_size,
                 target_vocab_size,
                 batch_size=32,
                 embedding_size=300,
                 mode='train',
                 hidden_units=256,
                 depth=1,
                 beam_width=0,
                 cell_type='lstm',
                 dropout=0.2,
                 use_dropout=False,
                 use_residual=False,
                 optimizer='adam',
                 learning_rate=1e-3,
                 min_learning_rate=1e-6,
                 decay_steps=500000,
                 max_gradient_norm=5.0,
                 max_decode_step=None,
                 attention_type='Bahdanau',
                 bidirectional=False,
                 time_major=False,
                 seed=0,
                 parallel_iterations=None,
                 share_embedding=False,
                 pretrained_embedding=False):
        """
        :param input_vocab_size: 输入词表大小
        :param target_vocab_size: 输出词表大小
        :param batch_size: 数据batch大小
        :param embedding_size: 输入, 输出词表embedding的维度
        :param mode: mode类型(训练还是测试), train, 训练, decode, 预训练
        :param hidden_units: RNN 模型中间层大小, encoder和decoder层相同
        :param depth: RNN层数(深度), 深度越多, 速度越慢, 精度越高
        :param beam_width: beamsearch的超参数, 用于解码
        :param cell_type: rnn 神经元类型(lstm, gru)
        :param dropout: 随机丢弃数据比例(0-1之间), 防止过拟合
        :param use_dropout: 是否使用dropout
        :param use_residual: 是否使用residual
        :param optimizer: 使用哪个优化器, 随机梯度下降
        :param learning_rate: 学习率
        min_learning_rate: 最小学习率
        :param decay_steps: 衰减步数
        :param max_gradient_norm: 正则, 梯度, 裁剪系数
        :param max_decode_step: 最大decode 长度, 可以非常大
        :param attention_type: 使用attention类型
        :param bidirectional: 单向(False)还是双向, encoder
        :param time_major: 在计算过程中是不是以时间为主要批量数据
        :param seed: 层见操作随机数
        :param parallel_iterations: 并行执行RNN循环个数
        :param share_embedding: encoder和decoder是否共享
        :param pretrained_embedding: 是不是使用预训练的embedding
        """
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.depth = depth
        self.cell_type = cell_type.lower()
        self.use_dropout = use_dropout
        self.use_residual = use_residual
        self.attention_type = attention_type
        self.mode = mode
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.decay_steps = decay_steps
        self.max_gradient_norm = max_gradient_norm
        self.keep_prob = 1.0 - dropout
        self.bidirectional = bidirectional
        self.seed = seed
        self.pretrained_embedding = pretrained_embedding
        if isinstance(parallel_iterations, int):
            self.parallel_iterations = parallel_iterations
        else:  # if parallel_iterations is None:
            self.parallel_iterations = batch_size
        self.time_major = time_major
        self.share_embedding = share_embedding

        self.initializer = tf.random_uniform_initializer(
            -0.05, 0.05, dtype=tf.float32
        )
        # self.initializer = None

        assert self.cell_type in ('gru', 'lstm'), \
            'cell_type 应该是 GRU 或者 LSTM'

        if share_embedding:
            assert input_vocab_size == target_vocab_size, \
                '如果打开 share_embedding，两个vocab_size必须一样'

        assert mode in ('train', 'decode'), \
            'mode 必须是 "train" 或 "decode" 而不是 "{}"'.format(mode)

        assert dropout >= 0.0 and dropout < 1.0, '0 <= dropout < 1'

        assert attention_type.lower() in ('bahdanau', 'luong'), \
            '''attention_type 必须是 "bahdanau" 或 "luong" 而不是 "{}"
            '''.format(attention_type)

        assert beam_width < target_vocab_size, \
            'beam_width {} 应该小于 target vocab size {}'.format(
                beam_width, target_vocab_size
            )

        self.keep_prob_placeholder = tf.placeholder(
            tf.float32,
            shape=[],
            name='keep_prob'
        )

        self.global_step = tf.Variable(
            0, trainable=False, name='global_step'
        )

        self.use_beamsearch_decode = False
        self.beam_width = beam_width
        self.use_beamsearch_decode = True if self.beam_width > 0 else False
        self.max_decode_step = max_decode_step

        assert self.optimizer.lower() in \
               ('adadelta', 'adam', 'rmsprop', 'momentum', 'sgd'), \
            'optimizer 必须是下列之一： adadelta, adam, rmsprop, momentum, sgd'

        self.build_model()

    def build_model(self):
        """
        1. 初始化一个训练, 预测所需要的变量
        2. 构建编码器, (encoder)  build_encoder -> encoder_cell -> build_single_cell
        3. 构建解码器(decoder)
        4. 构建优化器 (optimizer)
        5. 保存
        """
        self.init_placeholders()
        encoder_outputs, encoder_state = self.build_encoder()
        self.build_decoder(encoder_outputs, encoder_state)

        if self.mode == 'train':
            self.init_optimizer()

        self.saver = tf.train.Saver()

    def init_placeholders(self):
        """初始化一个训练, 预测所需要的变量"""
        self.add_loss = tf.placeholder(
            dtype=tf.float32,
            name='add_loss'
        )

        # 编码器输入，shape=(batch_size, time_step)
        # 有 batch_size 句话，每句话是最大长度为 time_step 的 index 表示
        self.encoder_inputs = tf.placeholder(
            dtype=tf.int32,
            shape=(self.batch_size, None),
            name='encoder_inputs'
        )

        # 编码器长度输入，shape=(batch_size, 1)
        # 指的是 batch_size 句话每句话的长度
        self.encoder_inputs_length = tf.placeholder(
            dtype=tf.int32,
            shape=(self.batch_size,),
            name='encoder_inputs_length'
        )

        if self.mode == 'train':
            # 训练模式

            # 解码器输入，shape=(batch_size, time_step)
            # 注意，会默认里面已经在每句结尾包含 <EOS>
            self.decoder_inputs = tf.placeholder(
                dtype=tf.int32,
                shape=(self.batch_size, None),
                name='decoder_inputs'
            )

            # 解码器输入的reward，用于强化学习训练，shape=(batch_size, time_step)
            self.rewards = tf.placeholder(
                dtype=tf.float32,
                shape=(self.batch_size, 1),
                name='rewards'
            )

            # 解码器长度输入，shape=(batch_size,)
            self.decoder_inputs_length = tf.placeholder(
                dtype=tf.int32,
                shape=(self.batch_size,),
                name='decoder_inputs_length'
            )

            self.decoder_start_token = tf.ones(
                shape=(self.batch_size, 1),
                dtype=tf.int32
            ) * WordSequence.START

            # 实际训练的解码器输入，实际上是 start_token + decoder_inputs
            self.decoder_inputs_train = tf.concat([
                self.decoder_start_token,
                self.decoder_inputs
            ], axis=1)

    def build_single_cell(self, n_hidden, use_residual):
        """
        构建一个单独的RNN cell
        :param n_hidden: 隐藏层神经元数量
        :param use_residual: 是否使用残差网络
        """
        if self.cell_type == 'gru':
            cell_type = GRUCell
        else:
            cell_type = LSTMCell

        cell = cell_type(n_hidden)

        if self.use_dropout:
            cell = DropoutWrapper(
                cell,
                dtype=tf.float32,
                output_keep_prob=self.keep_prob_placeholder,
                seed=self.seed
            )

        if use_residual:
            cell = ResidualWrapper(cell)

        return cell

    def build_encoder_cell(self):
        """
        构建单独的编码器cell
        :return:
        """
        return MultiRNNCell([
            self.build_single_cell(
                self.hidden_units,
                use_residual=self.use_residual
            )
            for _ in range(self.depth)
        ])

    def feed_embedding(self, sess, encoder=None, decoder=None):
        """加载预训练好的embedding
        """
        assert self.pretrained_embedding, \
            '必须开启pretrained_embedding才能使用feed_embedding'
        assert encoder is not None or decoder is not None, \
            'encoder 和 decoder 至少得输入一个吧大佬！'

        if encoder is not None:
            sess.run(self.encoder_embeddings_init,
                     {self.encoder_embeddings_placeholder: encoder})

        if decoder is not None:
            sess.run(self.decoder_embeddings_init,
                     {self.decoder_embeddings_placeholder: decoder})

    def build_encoder(self):
        """构建编码器"""
        # print("构建编码器")
        with tf.variable_scope('encoder'):
            # 构建 encoder_cell
            encoder_cell = self.build_encoder_cell()

            # 编码器的embedding
            with tf.device(_get_embed_device(self.input_vocab_size)):

                # 加载训练好的embedding
                if self.pretrained_embedding:

                    self.encoder_embeddings = tf.Variable(
                        tf.constant(
                            0.0,
                            shape=(self.input_vocab_size, self.embedding_size)
                        ),
                        trainable=True,
                        name='embeddings'
                    )
                    self.encoder_embeddings_placeholder = tf.placeholder(
                        tf.float32,
                        (self.input_vocab_size, self.embedding_size)
                    )
                    self.encoder_embeddings_init = \
                        self.encoder_embeddings.assign(
                            self.encoder_embeddings_placeholder)

                else:
                    self.encoder_embeddings = tf.get_variable(
                        name='embedding',
                        shape=(self.input_vocab_size, self.embedding_size),
                        initializer=self.initializer,
                        dtype=tf.float32
                    )

            # embedded之后的输入 shape = (batch_size, time_step, embedding_size)
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                params=self.encoder_embeddings,
                ids=self.encoder_inputs
            )

            if self.use_residual:
                self.encoder_inputs_embedded = \
                    layers.dense(self.encoder_inputs_embedded,
                                 self.hidden_units,
                                 use_bias=False,
                                 name='encoder_residual_projection')

            inputs = self.encoder_inputs_embedded
            if self.time_major:
                inputs = tf.transpose(inputs, (1, 0, 2))

            if not self.bidirectional:
                # 单向 RNN
                (
                    encoder_outputs,
                    encoder_state
                ) = tf.nn.dynamic_rnn(
                    cell=encoder_cell,
                    inputs=inputs,
                    sequence_length=self.encoder_inputs_length,
                    dtype=tf.float32,
                    time_major=self.time_major,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True
                )
            else:
                # 双向 RNN 比较麻烦
                encoder_cell_bw = self.build_encoder_cell()
                (
                    (encoder_fw_outputs, encoder_bw_outputs),
                    (encoder_fw_state, encoder_bw_state)
                ) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=encoder_cell,
                    cell_bw=encoder_cell_bw,
                    inputs=inputs,
                    sequence_length=self.encoder_inputs_length,
                    dtype=tf.float32,
                    time_major=self.time_major,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True
                )

                # 首先合并两个方向 RNN 的输出
                encoder_outputs = tf.concat(
                    (encoder_fw_outputs, encoder_bw_outputs), 2)

                encoder_state = []
                for i in range(self.depth):
                    encoder_state.append(encoder_fw_state[i])
                    encoder_state.append(encoder_bw_state[i])
                encoder_state = tuple(encoder_state)

            return encoder_outputs, encoder_state

    def build_decoder_cell(self, encoder_outputs, encoder_state):
        """
        构建解码器
        编码输出
        编码状态
        """
        encoder_inputs_length = self.encoder_inputs_length
        batch_size = self.batch_size

        if self.bidirectional:
            encoder_state = encoder_state[-self.depth:]

        if self.time_major:
            encoder_outputs = tf.transpose(encoder_outputs, (1, 0, 2))

        # 使用 BeamSearchDecoder 的时候，必须根据 beam_width 来成倍的扩大一些变量

        if self.use_beamsearch_decode:
            encoder_outputs = seq2seq.tile_batch(
                encoder_outputs, multiplier=self.beam_width)
            encoder_state = seq2seq.tile_batch(
                encoder_state, multiplier=self.beam_width)
            encoder_inputs_length = seq2seq.tile_batch(
                self.encoder_inputs_length, multiplier=self.beam_width)
            # 如果使用了 beamsearch 那么输入应该是 beam_width 倍于 batch_size 的
            batch_size *= self.beam_width

        # 下面是两种不同的 Attention 机制
        if self.attention_type.lower() == 'luong':
            # 'Luong' style attention: https://arxiv.org/abs/1508.04025
            self.attention_mechanism = LuongAttention(
                num_units=self.hidden_units,
                memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_length
            )
        else:  # Default Bahdanau
            # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
            self.attention_mechanism = BahdanauAttention(
                num_units=self.hidden_units,
                memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_length
            )

        # Building decoder_cell
        cell = MultiRNNCell([
            self.build_single_cell(
                self.hidden_units,
                use_residual=self.use_residual
            )
            for _ in range(self.depth)
        ])

        # 在非训练（预测）模式，并且没开启 beamsearch 的时候，打开 attention 历史信息
        alignment_history = (
                self.mode != 'train' and not self.use_beamsearch_decode
        )

        def cell_input_fn(inputs, attention):
            """根据attn_input_feeding属性来判断是否在attention计算前进行一次投影计算
            """
            if not self.use_residual:
                return array_ops.concat([inputs, attention], -1)

            attn_projection = layers.Dense(self.hidden_units,
                                           dtype=tf.float32,
                                           use_bias=False,
                                           name='attention_cell_input_fn')
            return attn_projection(array_ops.concat([inputs, attention], -1))

        cell = AttentionWrapper(
            cell=cell,
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.hidden_units,
            alignment_history=alignment_history,
            cell_input_fn=cell_input_fn,
            name='Attention_Wrapper')

        # 空状态
        decoder_initial_state = cell.zero_state(
            batch_size, tf.float32)

        # 传递encoder状态
        decoder_initial_state = decoder_initial_state.clone(
            cell_state=encoder_state)

        return cell, decoder_initial_state

    def build_decoder(self, encoder_outputs, encoder_state):
        """
        构建解码器
        :param encoder_outputs:
        :param encoder_state:
        :return:
        """
        with tf.variable_scope('decoder') as decoder_scope:
            (
                self.decoder_cell,
                self.decoder_initial_state
            ) = self.build_decoder_cell(encoder_outputs, encoder_state)

            # 解码器embedding
            with tf.device(_get_embed_device(self.target_vocab_size)):
                if self.share_embedding:
                    self.decoder_embeddings = self.encoder_embeddings
                elif self.pretrained_embedding:

                    self.decoder_embeddings = tf.Variable(
                        tf.constant(
                            0.0,
                            shape=(self.target_vocab_size,
                                   self.embedding_size)
                        ),
                        trainable=True,
                        name='embeddings'
                    )
                    self.decoder_embeddings_placeholder = tf.placeholder(
                        tf.float32,
                        (self.target_vocab_size, self.embedding_size)
                    )
                    self.decoder_embeddings_init = \
                        self.decoder_embeddings.assign(
                            self.decoder_embeddings_placeholder)
                else:
                    self.decoder_embeddings = tf.get_variable(
                        name='embeddings',
                        shape=(self.target_vocab_size, self.embedding_size),
                        initializer=self.initializer,
                        dtype=tf.float32
                    )

            self.decoder_output_projection = layers.Dense(
                self.target_vocab_size,
                dtype=tf.float32,
                use_bias=False,
                name='decoder_output_projection'
            )

            if self.mode == 'train':
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(
                    params=self.decoder_embeddings,
                    ids=self.decoder_inputs_train
                )
                inputs = self.decoder_inputs_embedded

                if self.time_major:
                    inputs = tf.transpose(inputs, (1, 0, 2))

                training_helper = seq2seq.TrainingHelper(
                    inputs=inputs,
                    sequence_length=self.decoder_inputs_length,
                    time_major=self.time_major,
                    name='training_helper'
                )

                # 训练的时候不在这里应用 output_layer
                # 因为这里会每个 time_step 的进行 output_layer 的投影计算，比较慢
                # 注意这个trick要成功必须设置 dynamic_decode 的 scope 参数
                training_decoder = seq2seq.BasicDecoder(
                    cell=self.decoder_cell,
                    helper=training_helper,
                    initial_state=self.decoder_initial_state,
                )

                # Maximum decoder time_steps in current batch
                max_decoder_length = tf.reduce_max(
                    self.decoder_inputs_length
                )


                (
                    outputs,
                    self.final_state, # contain attention
                    _ # self.final_sequence_lengths
                ) = seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=self.time_major,
                    impute_finished=True,
                    maximum_iterations=max_decoder_length,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True,
                    scope=decoder_scope
                )

                self.decoder_logits_train = self.decoder_output_projection(
                    outputs.rnn_output
                )

                # masks: masking for valid and padded time steps,
                # [batch_size, max_time_step + 1]
                self.masks = tf.sequence_mask(
                    lengths=self.decoder_inputs_length,
                    maxlen=max_decoder_length,
                    dtype=tf.float32, name='masks'
                )

                decoder_logits_train = self.decoder_logits_train
                if self.time_major:
                    decoder_logits_train = tf.transpose(decoder_logits_train,
                                                        (1, 0, 2))

                self.decoder_pred_train = tf.argmax(
                    decoder_logits_train, axis=-1,
                    name='decoder_pred_train')

                # 下面的一些变量用于特殊的学习训练
                # 自定义rewards，其实我这里是修改了masks
                # train_entropy = cross entropy
                self.train_entropy = \
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.decoder_inputs,
                        logits=decoder_logits_train)

                self.masks_rewards = self.masks * self.rewards

                self.loss_rewards = seq2seq.sequence_loss(
                    logits=decoder_logits_train,
                    targets=self.decoder_inputs,
                    weights=self.masks_rewards,
                    average_across_timesteps=True,
                    average_across_batch=True,
                )

                self.loss = seq2seq.sequence_loss(
                    logits=decoder_logits_train,
                    targets=self.decoder_inputs,
                    weights=self.masks,
                    average_across_timesteps=True,
                    average_across_batch=True,
                )

                self.loss_add = self.loss + self.add_loss

            elif self.mode == 'decode':
                # 预测模式，非训练

                start_tokens = tf.tile(
                    [WordSequence.START],
                    [self.batch_size]
                )
                end_token = WordSequence.END

                def embed_and_input_proj(inputs):
                    """输入层的投影层wrapper
                    """
                    return tf.nn.embedding_lookup(
                        self.decoder_embeddings,
                        inputs
                    )

                if not self.use_beamsearch_decode:
                    # Helper to feed inputs for greedy decoding:
                    # uses the argmax of the output
                    decoding_helper = seq2seq.GreedyEmbeddingHelper(
                        start_tokens=start_tokens,
                        end_token=end_token,
                        embedding=embed_and_input_proj
                    )
                    # Basic decoder performs greedy decoding at each time step
                    # print("building greedy decoder..")
                    inference_decoder = seq2seq.BasicDecoder(
                        cell=self.decoder_cell,
                        helper=decoding_helper,
                        initial_state=self.decoder_initial_state,
                        output_layer=self.decoder_output_projection
                    )
                else:
                    # Beamsearch is used to approximately
                    # find the most likely translation
                    # print("building beamsearch decoder..")
                    inference_decoder = BeamSearchDecoder(
                        cell=self.decoder_cell,
                        embedding=embed_and_input_proj,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=self.decoder_initial_state,
                        beam_width=self.beam_width,
                        output_layer=self.decoder_output_projection,
                    )

                if self.max_decode_step is not None:
                    max_decode_step = self.max_decode_step
                else:
                    # 默认 4 倍输入长度的输出解码
                    max_decode_step = tf.round(tf.reduce_max(
                        self.encoder_inputs_length) * 4)

                (
                    self.decoder_outputs_decode,
                    self.final_state,
                    _ # self.decoder_outputs_length_decode
                ) = (seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=self.time_major,
                    # impute_finished=True,	# error occurs
                    maximum_iterations=max_decode_step,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True,
                    scope=decoder_scope
                ))

                if not self.use_beamsearch_decode:

                    dod = self.decoder_outputs_decode
                    self.decoder_pred_decode = dod.sample_id

                    if self.time_major:
                        self.decoder_pred_decode = tf.transpose(
                            self.decoder_pred_decode, (1, 0))

                else:
                    self.decoder_pred_decode = \
                        self.decoder_outputs_decode.predicted_ids

                    if self.time_major:
                        self.decoder_pred_decode = tf.transpose(
                            self.decoder_pred_decode, (1, 0, 2))

                    self.decoder_pred_decode = tf.transpose(
                        self.decoder_pred_decode,
                        perm=[0, 2, 1])
                    dod = self.decoder_outputs_decode
                    self.beam_prob = dod.beam_search_decoder_output.scores

    def save(self, sess, save_path='model.ckpt'):
        """
        在TensorFlow里, 保存模型的格式有两种:
            ckpt: 训练模型后的保存, 这里面会保存所有的训练参数吗文件相对来说比较大,
                  可以用来进行模型的恢复和加载,
            pb: 用于模型最后的线上部署, 这里面的线上部署指的是TensorFlow Servinginking模型的发布, 一般发布成grpc形式的接口
        :param sess:
        :param save_path:
        :return:
        """
        self.saver.save(sess, save_path=save_path)

    def load(self, sess, save_path='model.ckpt'):
        print('try load model from ', save_path)
        self.saver.restore(sess=sess, save_path=save_path)

    def init_optimizer(self):
        """sgd, adadelta, adam(常用), rmsprop, momentum(常用)"""
        # 学习率下降算法
        learning_rate = tf.train.polynomial_decay(
            self.learning_rate,
            self.global_step,
            self.decay_steps,
            self.min_learning_rate,
            power=0.5
        )
        self.current_learning_rate = learning_rate

        # 设置优化器,合法的优化器如下
        # 'adadelta', 'adam', 'rmsprop', 'momentum', 'sgd'
        trainable_params = tf.trainable_variables()
        if self.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=learning_rate)
        elif self.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(
                learning_rate=learning_rate)
        elif self.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(
                learning_rate=learning_rate)
        elif self.optimizer.lower() == 'momentum':
            self.opt = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=0.9)
        elif self.optimizer.lower() == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)

        gradients = tf.gradients(self.loss, trainable_params)
        # Clip gradients by a given maximum_gradient_norm
        clip_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm)
        # Update the model
        self.updates = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params),
            global_step=self.global_step)

        # 使用包括rewards的loss进行更新
        # 是特殊学习的一部分
        gradients = tf.gradients(self.loss_rewards, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm)
        self.updates_rewards = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params),
            global_step=self.global_step)

        # 添加 self.loss_add 的 update
        gradients = tf.gradients(self.loss_add, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm)
        self.updates_add = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params),
            global_step=self.global_step)

    def check_feeds(self, encoder_inputs, encoder_inputs_length,
                    decoder_inputs, decoder_inputs_length, decode):
        """
        :param encoder_inputs: 一个整型的二维矩阵, [batch_size, max_source_time_steps]
        :param encoder_inputs_length: [batch_size], 每一个维度是encoder句子的真实长度
        :param decoder_inputs: 一个整型的二维矩阵, [batch_size, max_target_time_steps]
        :param decoder_inputs_length: [batch_size], 每一个维度是decoder句子的真实长度
        :param decode: 是训练模式train(decode=False), 还是预测模式decode(decode=True)
        :return: TensorFlow所需要的input_feed, 包含encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length
        """
        input_batch_size = encoder_inputs.shape[0]
        if input_batch_size != encoder_inputs_length.shape[0]:
            raise ValueError(
                "encoder_inputs和encoder_inputs_length的第一维度必须一致 "
                "这一维度是batch_size, %d != %d" % (
                    input_batch_size, encoder_inputs_length.shape[0]))

        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError(
                    "encoder_inputs和decoder_inputs的第一维度必须一致 "
                    "这一维度是batch_size, %d != %d" % (
                        input_batch_size, target_batch_size))
            if target_batch_size != decoder_inputs_length.shape[0]:
                raise ValueError(
                    "edeoder_inputs和decoder_inputs_length的第一维度必须一致 "
                    "这一维度是batch_size, %d != %d" % (
                        target_batch_size, decoder_inputs_length.shape[0]))

        input_feed = dict()

        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_inputs_length.name] = encoder_inputs_length

        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length

        return input_feed

    def train(self, sess, encoder_inputs, encoder_inputs_length,
              decoder_inputs, decoder_inputs_length,
              rewards=None, return_lr=False,
              loss_only=False, add_loss=None):
        """训练模型"""
        # 输入
        input_feed = self.check_feeds(
            encoder_inputs, encoder_inputs_length,
            decoder_inputs, decoder_inputs_length,
            False
        )

        # 设置 dropout
        input_feed[self.keep_prob_placeholder.name] = self.keep_prob

        if loss_only:
            # 输出
            return sess.run(self.loss, input_feed)

        if add_loss is not None:
            input_feed[self.add_loss.name] = add_loss
            output_feed = [
                self.updates_add, self.loss_add,
                self.current_learning_rate]
            _, cost, lr = sess.run(output_feed, input_feed)

            if return_lr:
                return cost, lr

            return cost

        if rewards is not None:
            input_feed[self.rewards.name] = rewards
            output_feed = [
                self.updates_rewards, self.loss_rewards,
                self.current_learning_rate]
            _, cost, lr = sess.run(output_feed, input_feed)

            if return_lr:
                return cost, lr
            return cost

        output_feed = [
            self.updates, self.loss,
            self.current_learning_rate]
        _, cost, lr = sess.run(output_feed, input_feed)

        if return_lr:
            return cost, lr

        return cost

    def get_encoder_embedding(self, sess, encoder_inputs):
        """获取经过embedding的encoder_inputs"""
        input_feed = {
            self.encoder_inputs.name: encoder_inputs
        }
        emb = sess.run(self.encoder_inputs_embedded, input_feed)
        return emb

    def entropy(self, sess, encoder_inputs, encoder_inputs_length,
                decoder_inputs, decoder_inputs_length):
        """获取针对一组输入输出的entropy
        相当于在计算P(target|source)
        """
        input_feed = self.check_feeds(
            encoder_inputs, encoder_inputs_length,
            decoder_inputs, decoder_inputs_length,
            False
        )
        input_feed[self.keep_prob_placeholder.name] = 1.0
        output_feed = [self.train_entropy, self.decoder_pred_train]
        entropy, logits = sess.run(output_feed, input_feed)
        return entropy, logits

    def predict(self, sess,
                encoder_inputs,
                encoder_inputs_length,
                attention=False):
        # 输入
        input_feed = self.check_feeds(encoder_inputs,
                                      encoder_inputs_length, None, None, True)

        input_feed[self.keep_prob_placeholder.name] = 1.0

        # Attention 输出
        if attention:
            assert not self.use_beamsearch_decode, \
                'Attention 模式不能打开 BeamSearch'

            pred, atten = sess.run([
                self.decoder_pred_decode,
                self.final_state.alignment_history.stack()
            ], input_feed)

            return pred, atten

        # BeamSearch 模式输出
        if self.use_beamsearch_decode:
            pred, beam_prob = sess.run([
                self.decoder_pred_decode, self.beam_prob
            ], input_feed)
            beam_prob = np.mean(beam_prob, axis=1)

            pred = pred[0]
            return pred

        # 普通（Greedy）模式输出
        print('output')
        print(self.decoder_pred_decode, input_feed)
        pred, = sess.run([self.decoder_pred_decode], input_feed)
        print(pred)
        return pred
