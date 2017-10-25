import tensorflow as tf
import tflearn

class vgg_reg_from_scratch:
    """ Fully Connected Network with depth-concatenation of inputs """
    def __init__(self):
        self.number_of_outputs = 1

    def net(self, inputList):
        # aggregate inputs in depth
        INPUT_CONCAT = tf.concat(axis=3, values=inputList)

        network = tflearn.conv_2d(INPUT_CONCAT, 64, 3, activation='relu')
        network = tflearn.conv_2d(network, 64, 3, activation='relu')
        network = tflearn.max_pool_2d(network, 2, strides=2)

        network = tflearn.conv_2d(network, 128, 3, activation='relu')
        network = tflearn.conv_2d(network, 128, 3, activation='relu')
        network = tflearn.max_pool_2d(network, 2, strides=2)

        network = tflearn.conv_2d(network, 256, 3, activation='relu')
        network = tflearn.conv_2d(network, 256, 3, activation='relu')
        network = tflearn.conv_2d(network, 256, 3, activation='relu')
        network = tflearn.max_pool_2d(network, 2, strides=2)

        network = tflearn.conv_2d(network, 512, 3, activation='relu')
        network = tflearn.conv_2d(network, 512, 3, activation='relu')
        network = tflearn.conv_2d(network, 512, 3, activation='relu')
        network = tflearn.max_pool_2d(network, 2, strides=2)

        network = tflearn.conv_2d(network, 512, 3, activation='relu')
        network = tflearn.conv_2d(network, 512, 3, activation='relu')
        network = tflearn.conv_2d(network, 512, 3, activation='relu')
        network = tflearn.max_pool_2d(network, 2, strides=2)

        network = tflearn.fully_connected(network, 4096, activation='relu')
        network = tflearn.dropout(network, 0.5)
        network = tflearn.fully_connected(network, 4096, activation='relu')
        network = tflearn.dropout(network, 0.5)
        # output is different from VGG
        network = tflearn.fully_connected(network, 1, activation='linear')

        return network
