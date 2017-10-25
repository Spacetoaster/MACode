import tensorflow as tf
import tflearn

class fc_rating_classification:
    """ Fully Connected Network with depth-concatenation of inputs """
    def __init__(self, number_of_outputs=100):
        self.number_of_outputs = number_of_outputs

    def net(self, inputList):
        # aggregate inputs
        INPUT_CONCAT = tf.concat(axis=3, values=inputList)

        net = tflearn.fully_connected(INPUT_CONCAT, 500, activation='relu')
        net = tflearn.fully_connected(net, 500, activation='relu')
        net = tflearn.fully_connected(net, 500, activation='relu')
        net = tflearn.fully_connected(net, self.number_of_outputs, activation="linear")
        return net
