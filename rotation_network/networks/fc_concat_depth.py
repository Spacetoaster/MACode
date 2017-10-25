import tensorflow as tf
import tflearn

class fc_concat_depth:
    """ Fully Connected Network with depth-concatenation of inputs """
    def __init__(self):
        self.number_of_outputs = 4

    def net(self, inputList):
        # aggregate inputs
        INPUT_CONCAT = tf.concat(axis=3, values=inputList)

        net = tflearn.fully_connected(INPUT_CONCAT, 500, activation='relu')
        net = tflearn.fully_connected(net, self.number_of_outputs, activation="linear")
        return net
