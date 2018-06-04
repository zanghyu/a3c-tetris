from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from constants import constants


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def cosineLoss(A, B, name):
    ''' A, B : (BatchSize, d) '''
    dotprod = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(A,1), tf.nn.l2_normalize(B,1)), 1)
    loss = 1-tf.reduce_mean(dotprod, name=name)
    return loss


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b


def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)




def universeHead(x, nConvs=4):
    ''' universe agent example
        input: [None, 42, 42, 1]; output: [None, 288];
    '''
    print('Using universe head design')
    x = tf.nn.elu(conv2d(x, 32, "lx1", [5, 5], [2, 2]))
    x = tf.nn.elu(conv2d(x, 64, "lx2", [5, 5], [2, 2]))
    print(x)
    x = flatten(x)
    return x


class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space, designHead='universe'):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space), name='x')
        size = 256
        x = universeHead(x)
        x = tf.nn.relu(linear(x, size, "li1", normalized_columns_initializer(0.01)))
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])

        # [0, :] means pick action of first state from batch. Hardcoded b/c
        # batch=1 during rollout collection. Its not used during batch training.
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.probs = tf.nn.softmax(self.logits, dim=-1)[0, :]

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    # def get_initial_features(self):
    #     # Call this function to get reseted lstm memory cells
    #     return self.state_init
    #
    def act(self, ob):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf],
                        {self.x: [ob]})
    #
    # def act_inference(self, ob, c, h):
    #     sess = tf.get_default_session()
    #     return sess.run([self.probs, self.sample, self.vf] + self.state_out,
    #                     {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})
    #
    def value(self, ob):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob]})[0]



