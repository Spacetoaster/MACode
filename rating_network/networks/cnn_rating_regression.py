import tensorflow as tf
import tflearn

class cnn_rating_regression:
    """ Regression-CNN with depth-concatenation of inputs, architecture like a smaller VGG """
    def __init__(self):
        self.number_of_outputs = 1
        self.dropout = 0.5

    def net(self, inputList):
        # aggregate inputs
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

        network = tflearn.fully_connected(network, 1024, activation='relu')
        network = tflearn.dropout(network, self.dropout)
        network = tflearn.fully_connected(network, 1024, activation='relu')
        network = tflearn.dropout(network, self.dropout)
        network = tflearn.fully_connected(network, 1, activation='linear')

        return network
