import inspect
import os

import numpy as np
import tensorflow as tf
import time
import tflearn

VGG_MEAN = [103.939, 116.779, 123.68]

class vgg_reg_small_pretrained:
    def __init__(self, vgg16_npy_path=None, number_of_outputs=100):
        if vgg16_npy_path is None:
            path = inspect.getfile(vgg_reg_small_pretrained)
            path = os.path.abspath(os.path.join(path, os.pardir))
            # npy muss in selben verzeichnis sein!
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")
        self.number_of_outputs = 1

    def net(self, inputList):
        # scale input to [244, 244, 5]
        rgb = inputList[0]
        rgb_upscaled = tf.image.resize_images(rgb, [224, 224])
        dl = inputList[1]
        dl_upscaled = tf.image.resize_images(dl, [224, 224])
        dt = inputList[2]
        dt_upscaled = tf.image.resize_images(dt, [224, 224])

        self.build(rgb_upscaled, dl_upscaled, dt_upscaled)
        return self.net

    def build(self, rgb, dl, dt):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        #rgb_scaled = rgb * 255.0
        rgb_scaled = rgb

        # Convert RGB to BGRq
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

        input = tf.concat(axis=3, values=[bgr, dl, dt])
        #input = bgr

        self.conv1_1 = self.conv_layer(input, "conv1_1", add_random_depth=True)
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.net = tflearn.conv_2d(self.pool3, 512, 3, activation='relu')
        self.net = tflearn.conv_2d(self.net, 512, 3, activation='relu')
        self.net = tflearn.conv_2d(self.net, 512, 3, activation='relu')
        self.net = tflearn.max_pool_2d(self.net, 2)

        self.net = tflearn.conv_2d(self.net, 512, 3, activation='relu')
        self.net = tflearn.conv_2d(self.net, 512, 3, activation='relu')
        self.net = tflearn.conv_2d(self.net, 512, 3, activation='relu')
        self.net = tflearn.max_pool_2d(self.net, 2, strides=2)

        # fully connected layers replaced and output is single neuron
        self.net = tflearn.fully_connected(self.net, 4096, activation='relu')
        self.net = tflearn.dropout(self.net, 0.5)
        self.net = tflearn.fully_connected(self.net, 4096, activation='relu')
        self.net = tflearn.dropout(self.net, 0.5)
        self.net = tflearn.fully_connected(self.net, 1, activation='linear')

        print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name, add_random_depth=False):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name, add_random_depth)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name, add_random_depth=False):
        filter = self.data_dict[name][0]
        if add_random_depth:
            depth_filter_shape = (filter.shape[0], filter.shape[1], 2, filter.shape[3])
            depth_filter = tf.truncated_normal(depth_filter_shape)
            # shape should be [filter_height, filter_width, in_channels, out_channels]
            filter = tf.concat([filter, depth_filter], axis=2)
        return tf.Variable(filter, name="filter", trainable=True)

    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name="biases", trainable=True)

    def get_fc_weight(self, name):
        return tf.Variable(self.data_dict[name][0], name="weights", trainable=True)
