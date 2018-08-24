
import tensorflow as tf
import numpy as np


class NetworkFactory:
    def NormDistFactory(data, dimension, complexity, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(data, complexity, tf.nn.relu, name = 'HiddenLayer', trainable=trainable)
            mu = 2 * tf.layers.dense(l1, dimension, tf.nn.tanh, name = 'mu', trainable=trainable)
            sigma = tf.layers.dense(l1, dimension, tf.nn.softplus, name = 'sigma', trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), act_type = 'relu',last=False):
    # 只有在一个build block 输出的时候才有activation 操作
        conv = mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
        if last:
            return bn
        else:
            bn = mx.symbol.BatchNorm(data=conv)
            act = mx.symbol.Activation(data=bn, act_type=act_type)
            return act

    def ResidualFactory(data, num_filter, diff_dim=False):
        if diff_dim:
            conv1 = ConvFactory(data=data,  num_filter=num_filter[0], kernel=(3,3), stride=(2,2), pad=(1,1), last=False)
            conv2 = ConvFactory(data=conv1, num_filter=num_filter[1], kernel=(3,3), stride=(1,1), pad=(1,1), last=True)
            # 输入的build block的维数和当前build block的维数不同, 所以, 使用conv操作来进行维数匹配
            _data = mx.symbol.Convolution(data=data,  num_filter=num_filter[1], kernel=(3,3), stride=(2,2), pad=(1,1))
            data  = _data+conv2
            bn    = mx.symbol.BatchNorm(data=data)
            act   = mx.symbol.Activation(data=bn, act_type='relu')
            return act
        else:
            _data=data
            conv1 = ConvFactory(data=data,  num_filter=num_filter[0], kernel=(3,3), stride=(1,1), pad=(1,1), last=False)
            conv2 = ConvFactory(data=conv1, num_filter=num_filter[1], kernel=(3,3), stride=(1,1), pad=(1,1), last=True)
            data  = _data+conv2
            bn    = mx.symbol.BatchNorm(data=data)
            act   = mx.symbol.Activation(data=bn, act_type='relu')
            return act