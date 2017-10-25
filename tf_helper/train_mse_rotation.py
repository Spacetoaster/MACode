import os
import tensorflow as tf
import math
from network import Network


class MSERegressionTrainerRotation():
    """ Trains a given network model with the data of the passed readers and saves the model to the given path """

    def __init__(self, reader_train, reader_test, num_epochs, num_train, num_test, save_path, network,
                 learning_rate=0.001, accuracy_error=25, batch_size=10, pretrained=None, combination=False):
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
        self.pretrained = pretrained
        self.combination = combination

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
        def axis_angle_from_quaternion(quaternion):
            qx = tf.slice(quaternion, [0, 1], [batch_size, 1])
            qy = tf.slice(quaternion, [0, 2], [batch_size, 1])
            qz = tf.slice(quaternion, [0, 3], [batch_size, 1])
            qw = tf.slice(quaternion, [0, 0], [batch_size, 1])
            # normalize quaternion
            q_length = tf.sqrt(tf.reduce_sum([tf.pow(qw, 2), tf.pow(qx, 2), tf.pow(qy, 2), tf.pow(qz, 2)], 0))
            qw = tf.divide(qw, q_length)
            qx = tf.divide(qx, q_length)
            qy = tf.divide(qy, q_length)
            qz = tf.divide(qz, q_length)
            normalized_quaternion = tf.stack([qw, qx, qy, qz])
            # calculate angle and axis
            angle = tf.divide(tf.multiply(tf.multiply(2.0, tf.acos(qw)), 180.0), math.pi)
            axis_x = tf.divide(qx, tf.sqrt(tf.subtract(1.0, tf.multiply(qw, qw))))
            axis_y = tf.divide(qy, tf.sqrt(tf.subtract(1.0, tf.multiply(qw, qw))))
            axis_z = tf.divide(qz, tf.sqrt(tf.subtract(1.0, tf.multiply(qw, qw))))
            return angle, axis_x, axis_y, axis_z, normalized_quaternion

        rangle, rx, ry, rz, rn = axis_angle_from_quaternion(net)
        langle, lx, ly, lz, ln = axis_angle_from_quaternion(label)
        # accuracy for quaternion angle
        angle_diff = tf.abs(tf.subtract(rangle, langle))
        correct_angle = tf.less_equal(angle_diff, self.accuracy_error)
        accuracy_angle = tf.reduce_mean(tf.cast(correct_angle, tf.float32))

        # accuracy for quaternion axis
        axis_angle_radians = tf.acos(tf.clip_by_value(tf.reduce_sum([tf.multiply(rx, lx), tf.multiply(ry, ly), tf.multiply(rz, lz )], 0), -1, 1))
        axis_angle_degrees = tf.divide(tf.multiply(axis_angle_radians, 180.0), math.pi)
        correct_axis = tf.less_equal(axis_angle_degrees, self.accuracy_error)
        accuracy_axis = tf.reduce_mean(tf.cast(correct_axis, tf.float32))

        # accuracys for both and stacked
        accuracy_both = tf.reduce_mean(tf.cast(tf.logical_and(correct_angle, correct_axis), tf.float32))
        accuracy = tf.stack([accuracy_angle, accuracy_axis, accuracy_both])

        # accuracy summaries
        accuracy_angle_placeholder = tf.placeholder(dtype=tf.float32)
        accuracy_angle_summary = tf.summary.scalar('Accuracy Angle', accuracy_angle_placeholder)
        accuracy_axis_placeholder = tf.placeholder(dtype=tf.float32)
        accuracy_axis_summary = tf.summary.scalar('Accuracy Axis', accuracy_axis_placeholder)
        accuracy_both_placeholder = tf.placeholder(dtype=tf.float32)
        accuracy_both_summary = tf.summary.scalar('Accuracy Both', accuracy_both_placeholder)

        # init and session
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        saver = tf.train.Saver()
        if self.combination:
            rating_variables = [v for v in tf.trainable_variables() if 'rating' in v.name]
            loader = tf.train.Saver(network_model.graph.getRatingDict(rating_variables))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        def writeSummary(writer, summary_op, dict, step):
            summary = sess.run(summary_op, dict)
            writer.add_summary(summary, step)

        def train_batch():
            global cost, train_loss, avg_batch_cost
            # load training batch
            inputs_batch, labels_batch = network_model.getTrainBatch(sess)
            feed_dict = network_model.getFeedDict(inputs_batch, labels_batch)
            # train and calculate loss
            _, cost = sess.run([train_op, loss], feed_dict)
            train_loss = sess.run(loss_summary, feed_dict={loss_placeholder: cost})
            train_batch_accuracys = sess.run(accuracy, feed_dict=feed_dict)
            return train_batch_accuracys

        def run_validation_set():
            total_test_batch = int(num_test / batch_size)
            costs = []
            accuracys = []
            angle_diffs = []
            axis_diffs = []
            for step in range(total_test_batch):
                # load validation batch
                inputs_batch, labels_batch = network_model.getTestBatch(sess)
                feed_dict = network_model.getFeedDict(inputs_batch, labels_batch)
                cost_validation = sess.run(loss, feed_dict)
                costs.append(cost_validation)
                angle_diff_batch, axis_diff_batch = sess.run([angle_diff, axis_angle_degrees], feed_dict)
                angle_diffs.append(angle_diff_batch)
                axis_diffs.append(axis_diff_batch)
                # accuracys
                valid_batch_accuracys = sess.run(accuracy, feed_dict=feed_dict)
                accuracys.append(valid_batch_accuracys)
            cost_avg = tf.cast(tf.reduce_mean(costs), tf.float32).eval()
            validation_loss = sess.run(loss_summary, feed_dict={loss_placeholder: cost_avg})
            accu_avg = tf.cast(tf.reduce_mean(tf.stack(accuracys), 0), tf.float32).eval()
            # tensorboard
            test_writer.add_summary(validation_loss, global_step)
            writeSummary(test_writer, accuracy_angle_summary, {accuracy_angle_placeholder: accu_avg[0]}, global_step)
            writeSummary(test_writer, accuracy_axis_summary, {accuracy_axis_placeholder: accu_avg[1]}, global_step)
            writeSummary(test_writer, accuracy_both_summary, {accuracy_both_placeholder: accu_avg[2]}, global_step)
            # print avg angle and axis
            avgAxis = tf.cast(tf.reduce_mean(tf.stack(axis_diffs)), tf.float32).eval()
            avgAngle = tf.cast(tf.reduce_mean(tf.stack(angle_diffs)), tf.float32).eval()
            print("Avg-Angle: {0}, Avg-Axis: {1}".format(avgAngle, avgAxis))
            # test_writer.add_summary(validation_accuracy, global_step)
            print("Test-Loss: {0:10.8f}, Test-Accuracy (Angle, Axis, Both): {1:2.2f}, {2:2.2f}, {3:2.2f} ".format(
                cost_validation, accu_avg[0], accu_avg[1], accu_avg[2]))
            return accu_avg[2]

        def save_model():
            saved_path = saver.save(sess, os.path.join(modelPath, "model.ckpt"))
            print("Model saved in file: ", saved_path)

        with tf.Session(config=config) as sess:
            # tensorboard writers
            train_writer = tf.summary.FileWriter(os.path.join(logPath, "train"), graph=sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(logPath, "test"))
            sess.run(init_op)
            print("init done")
            if self.pretrained:
                ckpt = tf.train.get_checkpoint_state(os.path.join(self.pretrained, "model"))
                if ckpt and ckpt.model_checkpoint_path:
                    if not self.combination:
                        saver.restore(sess, ckpt.model_checkpoint_path)
                    else:
                        loader.restore(sess, ckpt.model_checkpoint_path)
                    print("Restored Model")
                else:
                    print("Could not restore model!")
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
                # calcuate averages over train batch
                avg_batch_cost /= total_batch
                accu_avg = tf.cast(tf.reduce_mean(tf.stack(accuracys), 0), tf.float32).eval()
                # write train summaries
                writeSummary(train_writer, accuracy_angle_summary, {accuracy_angle_placeholder: accu_avg[0]}, global_step)
                writeSummary(train_writer, accuracy_axis_summary, {accuracy_axis_placeholder: accu_avg[1]}, global_step)
                writeSummary(train_writer, accuracy_both_summary, {accuracy_both_placeholder: accu_avg[2]}, global_step)
                # train_accuracy = sess.run(accuracy_summary, feed_dict={accuracy_placeholder: accu_avg})
                # train_writer.add_summary(train_accuracy, global_step)
                print("Epoch {0:5} of {1:5} ### Train-Loss: {2:10.8f} ### Train-Accuracy (Angle, Axis, Both): ".format(
                    epoch + 1, num_epochs, avg_batch_cost) + "{0:2.2f}, {1:2.2f}, {2:2.2f}".format(accu_avg[0],
                    accu_avg[1], accu_avg[2]))
                # validation
                acc_valid = run_validation_set()
                if acc_valid >= previous_best:
                    previous_best = acc_valid
                    save_model()

            coord.request_stop()
            coord.join(threads)
            sess.close()
