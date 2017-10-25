""" prediction with a trained multiview-model on a tfrecords-file """

import argparse
import tensorflow as tf
import quaternion_reader
from networks import *
from tf_helper.network import Network
import sklearn.metrics
import tf_helper.quaternion


def parseArguments():
    parser = argparse.ArgumentParser(description='Predict on trained network')
    parser.add_argument('predict_data', type=str, help='path to test data (tf-records file)')
    parser.add_argument('--num_samples', default=100, type=int, help='number of samples for prediction')
    parser.add_argument('--input_format', nargs='+', type=int, default=[3, 100, 100, 3],
                        help='input format: [num_images, width, height, depth]')
    parser.add_argument('--nntype', type=str, default='someNetwork', help="neural network to use")
    parser.add_argument('--withName', dest='withName', action='store_true', default=False)
    return parser.parse_args()

def main():
    args = parseArguments()
    predict(args.predict_data, args.num_samples, args.input_format, args.nntype, args.withName)

def predict(predict_data, num_samples=100, input_format=[3, 100, 100, 3], nntype="someNetwork", withName=False):
    records_file = predict_data

    try:
        network_model = getattr(globals()[nntype], nntype)()
    except KeyError:
        print "Network does not exist!"
        return
    avg_loss = 0
    num_samples = num_samples

    graph = network_model
    test_reader = quaternion_reader.QuaternionReader(predict_data, number_of_images=input_format[0],
                                                     image_width=input_format[1],
                                                     image_height=input_format[2],
                                                     image_depth=input_format[3], batch_size=1, withName=withName)
    network_model = Network(test_reader, test_reader, graph)

    print("Predicting [num_samples={0}]".format(num_samples))
    print("predict_data: " + records_file)

    net = network_model.output()

    loss = tf.reduce_mean(tf.square(tf.subtract(net, network_model.label)))

    resultAngle, resultX, resultY, resultZ, _ = tf_helper.quaternion.axis_angle_from_quaternion(net, 1)
    labelAngle, labelX, labelY, labelZ, _ = tf_helper.quaternion.axis_angle_from_quaternion(network_model.label, 1)
    #
    accuracies_25, angle_diff, axis_diff = tf_helper.quaternion.accuracies(resultAngle, resultX, resultY, resultZ,
                                                                        labelAngle, labelX, labelY, labelZ, 25)
    accuracies_20, _, _ = tf_helper.quaternion.accuracies(resultAngle, resultX, resultY, resultZ,
                                                                        labelAngle, labelX, labelY, labelZ, 20)
    accuracies_15, _, _ = tf_helper.quaternion.accuracies(resultAngle, resultX, resultY, resultZ,
                                                          labelAngle, labelX, labelY, labelZ, 15)
    accuracies_10, _, _ = tf_helper.quaternion.accuracies(resultAngle, resultX, resultY, resultZ,
                                                          labelAngle, labelX, labelY, labelZ, 10)
    # Init Ops and Config
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    def predict_sample():
        inputs, label = network_model.getTrainBatch(sess)
        feed_dict = network_model.getFeedDict(inputs, label)
        prediction_result, prediction_loss = sess.run([net, loss], feed_dict)
        return prediction_result, label, prediction_loss, feed_dict

    def predict_sample_with_name():
        inputs, label, name = network_model.getTrainBatchWithName(sess)
        feed_dict = network_model.getFeedDict(inputs, label)
        prediction_result, prediction_loss = sess.run([net, loss], feed_dict)
        return prediction_result, label, prediction_loss, feed_dict, name

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        print("init done")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        ckpt = tf.train.get_checkpoint_state("./model")
        if ckpt and ckpt.model_checkpoint_path:
            print("Restored Model")
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Could not restore model!")

        return_predictions = []
        return_labels = []
        return_names = []

        accuracies_list_25 = []
        accuracies_list_20 = []
        accuracies_list_15 = []
        accuracies_list_10 = []
        for i in range(num_samples):
            if not withName:
                result, label, my_loss, dict = predict_sample()
            else:
                result, label, my_loss, dict, liverName = predict_sample_with_name()
            return_predictions.append(result[0])
            return_labels.append(label[0])
            if withName:
                return_names.append(liverName[0])
            # angle and axis difference
            rangle, rx, ry, rz = sess.run([resultAngle, resultX, resultY, resultZ], feed_dict=dict)
            langle, lx, ly, lz = sess.run([labelAngle, labelX, labelY, labelZ], feed_dict=dict)
            predicted_angle_diff = sess.run(angle_diff, feed_dict=dict)
            predicted_axis_diff = sess.run(axis_diff, feed_dict=dict)
            # accuracies
            accu_25, accu_20, accu_15, accu_10 = sess.run([accuracies_25, accuracies_20, accuracies_15, accuracies_10], feed_dict=dict)
            accuracies_list_25.append(accu_25)
            accuracies_list_20.append(accu_20)
            accuracies_list_15.append(accu_15)
            accuracies_list_10.append(accu_10)
            avg_loss += my_loss
            print(
                "r: {0} ({1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}) | l: {5} ({6:.2f}, {7:.2f}, {8:.2f}, {9:.2f}) | loss: {10:.2f} | diff(deg/axis): {11:.2f} {12:.2f}".format(
                    result[0], rangle[0][0], rx[0][0], ry[0][0], rz[0][0], label[0], langle[0][0], lx[0][0], ly[0][0],
                    lz[0][0], my_loss, predicted_angle_diff[0][0], predicted_axis_diff[0][0]))
            if withName:
                print("name: {0}".format(liverName))
        # avg loss
        avg_loss /= num_samples
        # avg accuracies
        accu_avg_25 = tf.cast(tf.reduce_mean(tf.stack(accuracies_list_25), 0), tf.float32).eval()
        accu_avg_20 = tf.cast(tf.reduce_mean(tf.stack(accuracies_list_20), 0), tf.float32).eval()
        accu_avg_15 = tf.cast(tf.reduce_mean(tf.stack(accuracies_list_15), 0), tf.float32).eval()
        accu_avg_10 = tf.cast(tf.reduce_mean(tf.stack(accuracies_list_10), 0), tf.float32).eval()

        print("accuracy [angle | axis | both] (error: 25 deg): {0:.2f} | {1:.2f} | {2:.2f}".format(accu_avg_25[0],
                                                                                    accu_avg_25[1], accu_avg_25[2]))
        print("accuracy [angle | axis | both] (error: 20 deg): {0:.2f} | {1:.2f} | {2:.2f}".format(accu_avg_20[0],
                                                                                    accu_avg_20[1],accu_avg_20[2]))
        print("accuracy [angle | axis | both] (error: 15 deg): {0:.2f} | {1:.2f} | {2:.2f}".format(accu_avg_15[0],
                                                                                    accu_avg_15[1], accu_avg_15[2]))
        print("accuracy [angle | axis | both] (error: 10 deg): {0:.2f} | {1:.2f} | {2:.2f}".format(accu_avg_10[0],
                                                                                    accu_avg_10[1], accu_avg_10[2]))
        coord.request_stop()
        coord.join(threads)
        sess.close()

        return return_predictions, return_labels, return_names


if __name__ == '__main__': main()
