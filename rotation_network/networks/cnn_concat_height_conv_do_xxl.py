import tensorflow as tf
import tflearn

class cnn_concat_height_conv_do_xxl:
    """ CNN with Depth-Concatenation of Inputs and Dropout at the FC-Layer """
    def __init__(self, fc_multiplicator=1, conv_multiplicator=1, dropout=0.9):
        self.number_of_outputs = 4
        self.fc_mult = fc_multiplicator
        self.conv_mult = conv_multiplicator
        self.dropout = dropout

    def input_layer(self, INPUT):
        net_layer = tflearn.conv_2d(INPUT, 16 * self.conv_mult, 5, activation='relu')
        net_layer = tflearn.max_pool_2d(net_layer, 2)
        #net_layer = tflearn.dropout(net_layer, 0.8)

        net_layer = tflearn.conv_2d(net_layer, 32 * self.conv_mult, 3, activation='relu')
        net_layer = tflearn.max_pool_2d(net_layer, 2)
        #net_layer = tflearn.dropout(net_layer, 0.8)

        net_layer = tflearn.conv_2d(net_layer, 64 * self.conv_mult, 3, activation='relu')
        net_layer = tflearn.max_pool_2d(net_layer, 2)
        #net_layer = tflearn.dropout(net_layer, 0.8)

        net_layer = tflearn.conv_2d(net_layer, 128 * self.conv_mult, 3, activation='relu')
        net_layer = tflearn.max_pool_2d(net_layer, 2)

        return net_layer


    def net(self, inputList):
        nets = []
        for input in inputList:
            nets.append(self.input_layer(input))

        # aggregate outputs of networks
        INPUT_CONCAT = tf.concat(axis=1, values=nets)

        net = tflearn.conv_2d(INPUT_CONCAT, 128 * self.conv_mult, 3, activation='relu')
        net = tflearn.max_pool_2d(net, 2)
        net = tflearn.conv_2d(net, 256 * self.conv_mult, 3, activation='relu')
        net = tflearn.max_pool_2d(net, 2)
        net = tflearn.conv_2d(net, 512 * self.conv_mult, 3, activation='relu')
        net = tflearn.conv_2d(net, 512 * self.conv_mult, 3, activation='relu')
        net = tflearn.fully_connected(net, 150 * self.fc_mult, activation='relu')
        net = tflearn.dropout(net, self.dropout)
        net = tflearn.fully_connected(net, 100 * self.fc_mult, activation='relu')
        net = tflearn.dropout(net, self.dropout)
        net = tflearn.fully_connected(net, self.number_of_outputs, activation='linear')
        return net
