import tensorflow as tf
import tflearn

class vgg_rating_classification:
    """ Architecture of VGG Net with softmax output (Classification) """
    def __init__(self, number_of_outputs=100):
        self.number_of_outputs = number_of_outputs

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
        # use no softmax because of training with logits!
        # add softmax in prediction!
        # changed to 100 Outputs!
        network = tflearn.fully_connected(network, self.number_of_outputs, activation='linear')

        return network
