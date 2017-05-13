########################################################################################
# Guan Heng, 2017.5.10                                                                 #
# trainable VGG16 implementation in TensorFlow                                         #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://github.com/machrisaa/tensorflow-vgg                               #
# Weights from https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM  #
########################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re

__author__ = "GuanHeng"


class Net(object):
    """Base Net class
    """
    def __init__(self, model_npy_path=None, trainable=True, dropout=0.5):
        if model_npy_path is not None:
            self.data_dict = np.load(model_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout

    def avg_pooling(self, bottom, kernel_size, strides=2, name=None):
        """subsample a convolution layer using average pool rule

        :param bottom:
        :param kernel_size:
        :param strides:
        :param name:
        :return:
        """
        return tf.nn.avg_pool(bottom,
                              ksize=[1, kernel_size[0], kernel_size[1], 1],
                              strides=[1, strides, strides, 1],
                              padding='SAME', name=name)

    def max_pooling(self, bottom, kernel_size, strides=2, name=None):
        return tf.nn.max_pool(bottom,
                              ksize=[1, kernel_size[0], kernel_size[1], 1],
                              strides=[1, strides, strides, 1],
                              padding='SAME', name=name)

    def conv2d(self, bottom, kernel_shape, name, strides=1):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(kernel_shape, name)

            conv = tf.nn.conv2d(bottom, filt, [1, strides, strides, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def conv2d_basic(self, bottom, kernel_shape, name, strides=1):
        """a basic 2-dimentional convolution layer

        :param bottom:
        :param kernel_shape:
        :param name:
        :param strides:
        :return:
        """
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(kernel_shape, name)

            conv = tf.nn.conv2d(bottom, filt, [1, strides, strides, 1], padding='SAME')
            return tf.nn.bias_add(conv, conv_biases)

    def fc_layer(self, bottom, in_size, out_size, name):
        """ full connected layer

        :param bottom:
        :param in_size:
        :param out_size:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def conv2d_transpose_strided(self, bottom, kernel_shape, output_shape=None, stride=2, name=None):
        """ deconvolution layer

        :param bottom:
        :param kernel_shape:
        :param output_shape:
        :param stride:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_trans_var(kernel_shape, name)
            if output_shape is None:
                output_shape = x.get_shape().as_list()
                output_shape[1] *= 2
                output_shape[2] *= 2
                output_shape[3] = W.get_shape().as_list()[2]
            # print output_shape
            conv = tf.nn.conv2d_transpose(bottom, filt, output_shape, strides=[1, stride, stride, 1], padding="SAME")
            return tf.nn.bias_add(conv, conv_biases)

    def get_conv_var(self, kernel_shape, name):
        initial_value = tf.truncated_normal(kernel_shape, 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([kernel_shape[3]], 0.0, 0.001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_conv_trans_var(self, kernel_shape, name):
        initial_value = tf.truncated_normal(kernel_shape, 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([kernel_shape[2]], 0.0, 0.001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], 0.0, 0.001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./model-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
