import tensorflow as tf


class Network:
    """ Class which represents the neural network model """

    def __init__(self, train_reader, test_reader, graph):
        self.train_reader = train_reader
        self.test_reader = test_reader
        self.number_of_inputs = train_reader.number_of_images
        self.graph = graph
        self.number_of_outputs = graph.number_of_outputs
        self.initializePlaceholders()
        self.initializeReaderInputs()

    def initializePlaceholders(self):
        assert self.train_reader.getInputList() == self.test_reader.getInputList(), "InputLists of Train/Test differ"
        inputList = self.train_reader.getInputList()
        self.inputs = []
        for input in inputList:
            self.inputs.append(tf.placeholder(shape=(None, input[0], input[1], input[2]), dtype=tf.float32))
        self.label = tf.placeholder(shape=(None, self.number_of_outputs), dtype=tf.float32)
        if self.train_reader.withName:
            self.name = tf.placeholder(shape=(None, self.number_of_outputs), dtype=tf.string)

    def initializeReaderInputs(self):
        if not self.train_reader.withName:
            self.images_train, self.labels_train = self.train_reader.inputs()
            self.images_test, self.labels_test = self.test_reader.inputs()
        else:
            self.images_train, self.labels_train, self.names_train = self.train_reader.inputs_with_name()
            self.images_test, self.labels_test, self.names_test  = self.test_reader.inputs_with_name()

    def getTrainBatch(self, sess):
        return sess.run([self.images_train, self.labels_train])

    def getTrainBatchWithName(self, sess):
        return sess.run([self.images_train, self.labels_train, self.names_train])

    def getTestBatch(self, sess):
        return sess.run([self.images_test, self.labels_test])

    def getTestBatchWithName(self, sess):
        return sess.run([self.images_test, self.labels_test, self.names_test])

    def output(self):
        return self.graph.net(self.inputs)

    def getFeedDict(self, inputs_batch, labels_batch):
        placeholderList = self.inputs + [self.label]
        dataList = inputs_batch + [labels_batch]
        assert len(placeholderList) == len(dataList), "Can't build feeddict! placeholders and data do not match!"
        feed_dict = {}
        for placeholder, data in zip(placeholderList, dataList):
            feed_dict[placeholder] = data
        return feed_dict

    # def getFeedDictWithName(self, inputs_batch, labels_batch, names_batch):
    #     placeholderList = self.inputs + [self.label] + [self.name]
    #     dataList = inputs_batch + [labels_batch] + [names_batch]
    #     assert len(placeholderList) == len(dataList), "Can't build feeddict! placeholders and data do not match!"
    #     feed_dict = {}
    #     for placeholder, data in zip(placeholderList, dataList):
    #         feed_dict[placeholder] = data
    #     return feed_dict
