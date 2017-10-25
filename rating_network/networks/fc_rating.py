import tensorflow as tf
import tflearn

class fc_rating:
    """ Fully Connected Network with depth-concatenation of inputs """
    def __init__(self):
        self.number_of_outputs = 1

    def net(self, inputList):
        # aggregate inputs
        INPUT_CONCAT = tf.concat(axis=3, values=inputList)

        net = tflearn.fully_connected(INPUT_CONCAT, 200, activation='relu')
        net = tflearn.fully_connected(net, 100, activation='relu')
        net = tflearn.fully_connected(net, 25, activation='relu')
        net = tflearn.fully_connected(net, self.number_of_outputs, activation="linear")
        return net
