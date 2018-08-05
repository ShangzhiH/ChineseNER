# encoding=utf-8
import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.rnn import LSTMCell, GRUCell, DropoutWrapper, MultiRNNCell
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode

from model_dataset import DatasetMaker
from eval_utils import entity_metric_collect

__all__ = ["TrainModel", "EvalModel"]


class BaseModel(object):
    def __init__(self, input_chars, flags, dropout):
        self.char_dim = flags.char_dim
        self.char_num = flags.char_num
        self.tag_num = flags.tag_num

        self.rnn_type = flags.rnn_type
        self.rnn_dim = flags.rnn_dim
        self.rnn_layer = flags.rnn_layer

        self.dropout = dropout
        self.initializer = xavier_initializer()

        self.input_chars = input_chars
        real_char = tf.sign(self.input_chars)
        self.char_len = tf.reduce_sum(real_char, reduction_indices=1)

    def build_graph(self):
        # embedding
        input_embedding = self._build_embedding_layer(self.input_chars)
        # rnn
        rnn_output = self._build_multilayer_rnn(input_embedding, self.rnn_type, self.rnn_dim, self.rnn_layer, self.char_len)
        # projection
        logits = self._build_projection_layer(rnn_output, self.tag_num)
        return logits

    def _build_embedding_layer(self, inputs):
        with tf.variable_scope("char_embedding"), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(name="char_embedding_lookup_table", shape=[self.char_num, self.char_dim])
        return tf.nn.embedding_lookup(self.char_lookup, inputs)

    def _create_rnn_cell(self, rnn_type, rnn_dim, rnn_layer):
        def _single_rnn_cell():
            single_cell = None
            if rnn_type == "LSTM":
                single_cell = LSTMCell(rnn_dim, initializer=self.initializer, use_peepholes=True)
            elif rnn_type == "GRU":
                single_cell = GRUCell(rnn_dim, kernel_initializer=self.initializer)
            cell = DropoutWrapper(single_cell, output_keep_prob=self.dropout)
            return cell
        multi_cell = MultiRNNCell([_single_rnn_cell() for _ in range(rnn_layer)])
        return multi_cell

    def _build_birnn_layer(self, rnn_input, rnn_type, rnn_dim, lengths):
        with tf.variable_scope("forward_rnn"), tf.device("/gpu:0"):
            forward_rnn_cell = self._create_rnn_cell(rnn_type, rnn_dim, 1)
        with tf.variable_scope("backward_rnn"), tf.device("/gpu:1"):
            backward_rnn_cell = self._create_rnn_cell(rnn_type, rnn_dim, 1)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(forward_rnn_cell, backward_rnn_cell, rnn_input, dtype=tf.float32,
                                                     sequence_length=lengths)
        return tf.concat(outputs, axis=2)

    def _build_multilayer_rnn(self, rnn_input, rnn_type, rnn_dim, rnn_layer, lengths):
        inputs = rnn_input
        for i in range(rnn_layer):
            with tf.variable_scope("Bi{} Sequence: Layer-{}".format(rnn_type, i+1)):
                inputs = self._build_birnn_layer(input, rnn_type, rnn_dim, lengths)
        return inputs

    def _build_projection_layer(self, inputs, output_dim):
        with tf.variable_scope("projection_layer"):
            projection_layer = tf.layers.Dense(units=output_dim, use_bias=False, kernel_initializer=self.initializer)
            logits = projection_layer.apply(inputs)
        return logits


class TrainModel(BaseModel):
    def __init__(self, iterator, flags):
        chars, tags = iterator.get_next()
        self.tags = tags
        super(TrainModel).__init__(chars, flags, flags.dropout)
        self.lr = flags.lr
        self.clip = flags.clip
        self.loss_type = flags.loss_type

        self.logits = self.build_graph()
        self.loss = self._build_loss_layer(self.logits, self.tags)
        self.train_op = self._optimizer(self.loss)
        self.saver = tf.train.Saver(tf.global_variables())

    def _build_loss_layer(self, inputs, tags):
        loss = None
        if self.loss_type == "softmax":
            loss = self._softmax_cross_entropy_loss(inputs, tags)
        elif self.loss_type == "crf":
            loss = self._crf_loss(inputs, tags, self.char_len)
        return loss

    @staticmethod
    def _softmax_cross_entropy_loss(project_logits, tags):
        with tf.variable_scope("softmax_cross_entropy_loss"):
            log_likelihood = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=project_logits, labels=tags)
            loss = tf.reduce_mean(log_likelihood)
        return loss

    def _crf_loss(self, project_logits, real_tags, lengths):
        batch_size = tf.shape(project_logits)[0]
        num_steps = tf.shape(project_logits)[1]
        with tf.variable_scope("crf_loss"):
            small = -1000.0
            # pad start position for crf loss
            start_logits = tf.concat([small * tf.ones(shape=[batch_size, 1, self.tag_num]),
                                      tf.zeros(shape=tf.zeros(batch_size, 1, 1))], axis=2)
            pad_logits = tf.cast(small * tf.ones([batch_size, num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=2)
            logits = tf.concat([start_logits, logits], axis=1)

            start_tags = tf.cast(self.tag_num * tf.ones([batch_size, 1]), tf.int32)
            tags = tf.concat([start_tags, real_tags], axis=1)

            trans = tf.get_variable("crf_transitions", shape=[self.tag_num + 1, self.tag_num + 1],
                                    initializer=self.initializer)
            log_likelihood, trans = crf_log_likelihood(inputs=logits, tag_indices=tags, sequence_lengths=lengths+1, transition_params=trans)
            loss = tf.reduce_mean(-log_likelihood)
        return loss

    def _optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(self.lr)
        grads_vars = optimizer.compute_gradients(loss)
        capped_grads_vars = [[tf.clip_by_value(g, -self.clip, self.clip), v] for g, v in grads_vars]
        train_op = optimizer.apply_gradients(capped_grads_vars)
        return train_op

    def train(self, session):
        loss_value, _ = session.run([self.loss, self.train_op])
        return loss_value


class EvalMode(BaseModel):
    def __init__(self, iterator, flags):
        chars, tags = iterator.get_next
        self.tags = tags
        super(EvalMode).__init__(chars, flags, 1.0)
        self.loss_type = flags.loss_type

        self.logits = self.build_graph()
        with tf.variable_scope("crf_loss"):
            self.trans = tf.get_variable("crf_transitions", shape=[self.tag_num + 1, self.tag_num + 1],
                                         initializer=self.initializer)

    def _logits_to_tag_ids(self, logits, matrix=None):
        predict_tag_ids = None
        if self.loss_type == "softmax":
            predict_tag_ids = np.argmax(logits, axis=2)
        elif self.loss_type == "crf":
            predict_tag_ids = self._decode(logits, self.char_len, matrix)
        return predict_tag_ids

    def _decode(self, logits, lengths, matrix):
        """
        apply viterbi decode to logits
        :param logits: [batch_size, num_steps, tag_num]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: crf transition matrix
        :return: path of the highest probability
        """
        paths = []
        small = -1000.0
        start = np.asarray([[small] * self.tag_num + [0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])

            pad_logit = np.concatenate([score, pad], axis=1)
            pad_logit = np.concatenate([start, pad_logit], axis=0)
            path, _ = viterbi_decode(pad_logit, matrix)
            paths.append(path[1:])
        return paths

    def evaluate(self, session):
        metric_dict = {}
        try:
            while True:
                predict_tag_ids = None
                if self.loss_type == "softmax":
                    logits = session.run(self.logits)
                    predict_tag_ids = self._logits_to_tag_ids(logits)
                elif self.loss_type == "crf":
                    logits, trans = session.run([self.logits, self.trans])
                    predict_tag_ids = self._logits_to_tag_ids(logits, trans)
                predict_tags = DatasetMaker.tag_ids_to_tags(predict_tag_ids)
                metric_dict = entity_metric_collect(self.tags, predict_tags, metric_dict)
        except tf.errors.OutOfRangeError:
            return metric_dict







