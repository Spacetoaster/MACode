import tensorflow as tf
import tflearn

class cnn_rating_classification_xs:
    """ Classification- CNN with depth-concatenation of inputs, architecture like a smaller VGG """
    def __init__(self, number_of_outputs=100):
        self.number_of_outputs = number_of_outputs

    def net(self, inputList):
        # aggregate inputs
        INPUT_CONCAT = tf.concat(axis=3, values=inputList)

        network = tflearn.conv_2d(INPUT_CONCAT, 64, 3, activation='relu')
        network = tflearn.conv_2d(network, 64, 3, activation='relu')
        network = tflearn.max_pool_2d(network, 2, strides=2)

        network = tflearn.conv_2d(network, 128, 3, activation='relu')
        network = tflearn.conv_2d(network, 128, 3, activation='relu')
        network = tflearn.max_pool_2d(network, 2, strides=2)

        network = tflearn.fully_connected(network, 128, activation='relu')
        network = tflearn.dropout(network, 0.5)
        network = tflearn.fully_connected(network, 128, activation='relu')
        network = tflearn.dropout(network, 0.5)
        network = tflearn.fully_connected(network, self.number_of_outputs, activation='linear')

        return network
