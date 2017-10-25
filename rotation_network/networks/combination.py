import tensorflow as tf
import tflearn

class combination:
    """ Regression-CNN with depth-concatenation of inputs, architecture like a smaller VGG """
    def __init__(self, fc_multiplicator=1, conv_multiplicator=1, dropout=0.5):
        self.dropout = 0.5
        self.number_of_outputs = 4

    def rating_net(self, inputList, reuse=False):
        # aggregate inputs
        INPUT_CONCAT = tf.concat(axis=3, values=inputList)

        network = tflearn.conv_2d(INPUT_CONCAT, 64, 3, activation='relu', scope="rating1", reuse=reuse)
        network = tflearn.conv_2d(network, 64, 3, activation='relu', scope="rating2", reuse=reuse)
        network = tflearn.max_pool_2d(network, 2, strides=2)

        network = tflearn.conv_2d(network, 128, 3, activation='relu', scope="rating3", reuse=reuse)
        network = tflearn.conv_2d(network, 128, 3, activation='relu', scope="rating4", reuse=reuse)
        network = tflearn.max_pool_2d(network, 2, strides=2)

        network = tflearn.conv_2d(network, 256, 3, activation='relu', scope="rating5", reuse=reuse)
        network = tflearn.conv_2d(network, 256, 3, activation='relu', scope="rating6", reuse=reuse)
        network = tflearn.conv_2d(network, 256, 3, activation='relu', scope="rating7", reuse=reuse)
        network = tflearn.max_pool_2d(network, 2, strides=2)

        return network


    def net(self, inputList):
        nets = []
        nets.append(self.rating_net(inputList[0], False))
        for input in inputList[1:]:
            nets.append(self.rating_net(input, True))

        # aggregate outputs of networks
        INPUT_CONCAT = tf.concat(axis=1, values=nets)

        net = tflearn.conv_2d(INPUT_CONCAT, 128, 3, activation='relu', scope="rotation1")
        net = tflearn.max_pool_2d(net, 2)
        net = tflearn.conv_2d(net, 256, 3, activation='relu', scope="rotation2")
        net = tflearn.conv_2d(net, 512, 3, activation='relu', scope="rotation3")
        net = tflearn.fully_connected(net, 150, activation='relu', scope="rotation4")
        net = tflearn.dropout(net, self.dropout)
        net = tflearn.fully_connected(net, 100, activation='relu', scope="rotation5")
        net = tflearn.dropout(net, self.dropout)
        net = tflearn.fully_connected(net, 4, activation='linear', scope="rotation6")
        return net

    def getRatingDict(self, rating_variables):
        reloadDict_rating = {
            'Conv2D/W': rating_variables[0],
            'Conv2D/b': rating_variables[1],
            'Conv2D_1/W': rating_variables[2],
            'Conv2D_1/b': rating_variables[3],
            'Conv2D_2/W': rating_variables[4],
            'Conv2D_2/b': rating_variables[5],
            'Conv2D_3/W': rating_variables[6],
            'Conv2D_3/b': rating_variables[7],
            'Conv2D_4/W': rating_variables[8],
            'Conv2D_4/b': rating_variables[9],
            'Conv2D_5/W': rating_variables[10],
            'Conv2D_5/b': rating_variables[11],
            'Conv2D_6/W': rating_variables[12],
            'Conv2D_6/b': rating_variables[13],
        }
        return reloadDict_rating