########################################################################################
# Guan Heng, 2017.5.10                                                                 #
# trainable VGG16 implementation in TensorFlow                                         #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://github.com/machrisaa/tensorflow-vgg                               #
# Weights from https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM  #
########################################################################################
import tensorflow as tf

import numpy as np
from functools import reduce
from net import Net

VGG_MEAN = [123.68, 116.779, 103.939]

NUM_OF_CLASS = 1000


class Vgg16(Net):
    """
    A trainable version VGG19.
    """

    def __init__(self, model_npy_path=None, trainable=True, dropout=0.9):
        super(Vgg16, self).__init__(model_npy_path, trainable, dropout)
        if model_npy_path is not None:
            self.data_dict = np.load(model_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout

    def build(self, rgb, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv2d(bgr, [3, 3, 3, 64], "conv1_1")
        self.conv1_2 = self.conv2d(self.conv1_1, [3, 3, 64, 64], "conv1_2")
        self.pool1 = self.max_pooling(self.conv1_2, [2, 2], name='pool1')

        self.conv2_1 = self.conv2d(self.pool1, [3, 3, 64, 128], "conv2_1")
        self.conv2_2 = self.conv2d(self.conv2_1, [3, 3, 128, 128], "conv2_2")
        self.pool2 = self.max_pooling(self.conv2_2, [2, 2], name='pool2')

        self.conv3_1 = self.conv2d(self.pool2, [3, 3, 128, 256], "conv3_1")
        self.conv3_2 = self.conv2d(self.conv3_1, [3, 3, 256, 256], "conv3_2")
        self.conv3_3 = self.conv2d(self.conv3_2, [3, 3, 256, 256], "conv3_3")
        self.pool3 = self.max_pooling(self.conv3_3, [2, 2], name='pool3')

        self.conv4_1 = self.conv2d(self.pool3, [3, 3, 256, 512], "conv4_1")
        self.conv4_2 = self.conv2d(self.conv4_1, [3, 3, 512, 512], "conv4_2")
        self.conv4_3 = self.conv2d(self.conv4_2, [3, 3, 512, 512], "conv4_3")
        self.pool4 = self.max_pooling(self.conv4_3, [2, 2], name='pool4')

        self.conv5_1 = self.conv2d(self.pool4, [3, 3, 512, 512], "conv5_1")
        self.conv5_2 = self.conv2d(self.conv5_1, [3, 3, 512, 512], "conv5_2")
        self.conv5_3 = self.conv2d(self.conv5_2, [3, 3, 512, 512], "conv5_3")
        self.pool5 = self.max_pooling(self.conv5_3, [2, 2], name='pool5')

        in_size = self.pool5.shape[1]*self.pool5.shape[2]*self.pool5.shape[3]
        print "full connect layer input size:", in_size
        # i can not use in_size based computing above to replace a number of 25088 here, otherwise,
        # if will output error "TypeError: Expected binary or unicode string, got Dimension(25088)",
        # i still figure out why it happened, if somebody solve it, please let me know it
        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer(self.relu7, 4096, 1000, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None

    def train_opts(self, labels, learning_rate):
        """return loss and train_opt of model
        """
        loss = tf.reduce_sum((self.prob - labels) ** 2)
        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        return train, loss

    def load_weights_npy(self, weight_file, sess):
        """load weights of .npy file of pretraining vgg16 model

        :param weight_file:
        :param sess:
        :return:
        """
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        # print 3, self.model_name[3], np.shape(weights[self.model_name[3]])
        for i, k in enumerate(keys):
            print i, k, weights[k].shape
            sess.run(self.parameters[i].assign(weights[k]))

    def load_weights_npz(self, weight_file, sess):
        """load weights of .npz file of pretraining vgg16 model

        :param weight_file:
        :param sess:
        :return:
        """
        weights = np.load(weight_file)
        param = weights['arr_0'].item()

        for i, k in enumerate(param):
            print i, k, param[k].shape
            sess.run(self.parameters[i].assign(param[k]))

    def save_weights_npz(self, sess, npy_path="./vgg16-save.npz"):
        """ save training weights as a .npz file

        :param sess:
        :param npy_path:
        :return:
        """
        assert isinstance(sess, tf.Session)

        data_dict = OrderedDict()

        param = sess.run(self.parameters)
        for i in range(len(param)):
            data_dict[self.model_name[i]] = param[i]

        np.savez(npy_path, data_dict)
        print "weights saved as path: ", npy_path
        return npy_path
