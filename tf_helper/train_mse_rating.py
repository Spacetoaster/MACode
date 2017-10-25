import os
import tensorflow as tf
from network import Network


class MSERegressionTrainerRating():
    """ Trains a given network model with the data of the passed readers and saves the model to the given path """

    def __init__(self, reader_train, reader_test, num_epochs, num_train, num_test, save_path, network,
                 learning_rate=0.001, accuracy_error=0.05, batch_size=10):
        self.reader_train = reader_train
        self.reader_test = reader_test
        self.num_epochs = num_epochs
        self.num_train = num_train
        self.num_test = num_test
        self.save_path = save_path
        self.network = network
        self.learning_rate = learning_rate
        self.accuracy_error = accuracy_error
        self.batch_size = batch_size

    def train(self):
        # parameters
        batch_size = self.batch_size
        learning_rate = self.learning_rate
        num_epochs = self.num_epochs
        num_train = self.num_train
        num_test = self.num_test

        # filepathes
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        logPath = os.path.join(self.save_path, "logs")
        modelPath = os.path.join(self.save_path, "model")
        if not os.path.exists(modelPath):
            os.makedirs(modelPath)

        print(
            "Training Network [num_epochs={0}, num_train={1}, num_test={2}, batch_size={3}]".format(num_epochs,
                                                                                                    num_train,
                                                                                                    num_test,
                                                                                                    batch_size))

        # graph of the network
        graph = self.network

        # network model
        network_model = Network(self.reader_train, self.reader_test, graph)
        net = network_model.output()
        label = network_model.label

        # regularization
        # vars = tf.trainable_variables()
        # l2_regularization_error = tf.add_n([tf.nn.l2_loss(v) for v in vars if '/W' in v.name])
        # l2_lambda = 0.0001

        # loss, train and validation functions
        mean_squared_error = tf.reduce_mean(tf.square(tf.subtract(net, label)))
        loss = mean_squared_error
        # loss_summary = tf.summary.scalar('MSE', loss)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        loss_placeholder = tf.placeholder(dtype=tf.float32)
        loss_summary = tf.summary.scalar('MSE', loss_placeholder)

        # accuracy
        correct_prediction = tf.less_equal(tf.abs(tf.subtract(tf.cast(net, tf.float32), tf.cast(label, tf.float32))),
                                           self.accuracy_error)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_placeholder = tf.placeholder(dtype=tf.float32)
        accuracy_summary = tf.summary.scalar('Accuracy', accuracy_placeholder)

        # init and session
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        def train_batch():
            global cost, train_loss, avg_batch_cost
            # load training batch
            inputs_batch, labels_batch = network_model.getTrainBatch(sess)
            feed_dict = network_model.getFeedDict(inputs_batch, labels_batch)
            # train and calculate loss
            _, cost = sess.run([train_op, loss], feed_dict)
            train_loss = sess.run(loss_summary, feed_dict={loss_placeholder: cost})
            train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
            return train_accuracy
            # cost = sess.run(loss, feed_dict={X: x_batch, Y: y_batch})

        def run_validation_set():
            total_test_batch = int(num_test / batch_size)
            costs = []
            accuracys = []
            for step in range(total_test_batch):
                # load validation batch
                inputs_batch, labels_batch = network_model.getTestBatch(sess)
                feed_dict = network_model.getFeedDict(inputs_batch, labels_batch)
                cost_validation = sess.run(loss, feed_dict)
                costs.append(cost_validation)
                accu = sess.run(accuracy, feed_dict)
                accuracys.append(accu)
            cost_avg = tf.cast(tf.reduce_mean(costs), tf.float32).eval()
            accu_avg = tf.cast(tf.reduce_mean(accuracys), tf.float32).eval()
            validation_loss = sess.run(loss_summary, feed_dict={loss_placeholder: cost_avg})
            validation_accuracy = sess.run(accuracy_summary, feed_dict={accuracy_placeholder: accu_avg})
            # tensorboard
            test_writer.add_summary(validation_loss, global_step)
            test_writer.add_summary(validation_accuracy, global_step)
            print("validation loss: {0:10.8f}, accuracy (error: {1:.2f}): {2:.2f} ".format(cost_validation,
                                                                                    self.accuracy_error, accu_avg))
            return accu_avg

        def save_model():
            saved_path = saver.save(sess, os.path.join(modelPath, "model.ckpt"))
            print("Model saved in file: ", saved_path)

        with tf.Session(config=config) as sess:
            # tensorboard writers
            train_writer = tf.summary.FileWriter(os.path.join(logPath, "train"), graph=sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(logPath, "test"))
            sess.run(init_op)
            print("init done")
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            global_step = 0
            previous_best = 0.0
            # run validation set for graph
            run_validation_set()
            for epoch in range(num_epochs):
                accuracys = []
                # training with BGD
                total_batch = int(num_train / batch_size)
                avg_batch_cost = 0
                for step in range(total_batch):
                    batch_accuracy = train_batch()
                    avg_batch_cost += cost
                    # tensorboard
                    global_step = epoch * total_batch + step
                    train_writer.add_summary(train_loss, global_step)
                    accuracys.append(batch_accuracy)
                avg_batch_cost /= total_batch
                accu_avg = tf.cast(tf.reduce_mean(accuracys), tf.float32).eval()
                train_accuracy = sess.run(accuracy_summary, feed_dict={accuracy_placeholder: accu_avg})
                train_writer.add_summary(train_accuracy, global_step)
                print(
                    "Epoch {0:5} of {1:5} ### Loss (Training Avg [Last Batch]): {2:10.8f} [{3:10.8f}]".format(epoch + 1,
                                                                                                              num_epochs,
                                                                                                              avg_batch_cost,
                                                                                                              cost))
                # validation
                acc_valid = run_validation_set()
                if acc_valid >= previous_best:
                    previous_best = acc_valid
                    save_model()

            coord.request_stop()
            coord.join(threads)
            sess.close()
