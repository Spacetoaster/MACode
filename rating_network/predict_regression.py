""" prediction with a trained rating-model (regression) on a tfrecords-file """

import argparse
import tensorflow as tf
import ratings_reader
from networks import *
from tf_helper.network import Network
import sklearn.metrics
import pickle


def parseArguments():
    parser = argparse.ArgumentParser(description='Predict on trained network')
    parser.add_argument('predict_data', type=str, help='path to test data (tf-records file)')
    parser.add_argument('--num_samples', default=100, type=int, help='number of samples for prediction')
    parser.add_argument('--nntype', type=str, default='fc_rating', help="neural network to use")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes to compare with classification")
    parser.add_argument('--no_depth', action='store_true', default=False)
    parser.add_argument('--restore_path', type=str, default="./model")
    parser.add_argument('--save_cm', type=str, default=None)
    return parser.parse_args()


def main():
    args = parseArguments()
    predict(args.predict_data, args.num_samples, args.nntype, args.num_classes, no_depth=args.no_depth,
            restorePath=args.restore_path, save_cm=args.save_cm)


def predict(predict_data, num_samples, nntype, num_classes, no_depth=False, restorePath="./model", save_cm=None):
    records_file = predict_data

    try:
        network_model = getattr(globals()[nntype], nntype)()
    except KeyError:
        print "Network does not exist!"
        return
    avg_loss = 0
    num_samples = num_samples

    graph = network_model
    test_reader = ratings_reader.RatingsReader(predict_data, batch_size=1, no_depth=no_depth)
    network_model = Network(test_reader, test_reader, graph)

    print("Predicting [num_samples={0}]".format(num_samples))
    print("predict_data: " + records_file)

    net = network_model.output()

    loss = tf.reduce_mean(tf.square(tf.subtract(net, network_model.label)))

    # accuracy
    prediction_errors = [0.05, 0.07, 0.1, 0.15, 0.20]
    prediction_error = tf.placeholder(dtype=tf.float32)
    correct_prediction = tf.less_equal(
        tf.abs(tf.subtract(tf.cast(net, tf.float32), tf.cast(network_model.label, tf.float32))), prediction_error)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    predictions = {key: [] for key in prediction_errors}

    # classification comparison
    reg_to_bins = tf.placeholder(dtype=tf.float32)
    bins = tf.clip_by_value(tf.cast(tf.floor(tf.multiply(reg_to_bins, num_classes)), tf.int32), 0,
                            num_classes - 1)

    # Init Ops and Config
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    def predict_sample(predictions):
        inputs, label = network_model.getTrainBatch(sess)
        feed_dict = network_model.getFeedDict(inputs, label)
        prediction_result, prediction_loss = sess.run([net, loss], feed_dict)
        for error in prediction_errors:
            feed_dict[prediction_error] = error
            predictions[error].append(sess.run(accuracy, feed_dict))
        return prediction_result, label, prediction_loss

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        print("init done")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        ckpt = tf.train.get_checkpoint_state(restorePath)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restored Model")
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Could not restore model!")

        regression_predictions = []
        labels = []
        for i in range(num_samples):
            result, label, my_loss = predict_sample(predictions)
            regression_predictions.append(result[0][0])
            labels.append(label[0][0])
            avg_loss += my_loss
            print("result: {0:.6f} \tlabel: {1:.6f} \tloss: {2:.6f}".format(result[0][0], label[0][0], my_loss))
        # loss
        avg_loss /= num_samples

        # regression accuracys
        predictions_output = ""
        for error in prediction_errors:
            predictions[error] = tf.cast(tf.reduce_mean(predictions[error]), tf.float32).eval()
            predictions_output += "accuracy ({0:.2f}): {1:.2f}\n".format(error, predictions[error])
        print predictions_output
        print("avg_loss: {0:.6f}".format(avg_loss))

        # classification accuracy
        return_predictions = regression_predictions
        return_labels = labels
        regression_predictions = bins.eval({reg_to_bins: regression_predictions})
        labels = bins.eval({reg_to_bins: labels})
        confusionMatrix = sklearn.metrics.confusion_matrix(labels, regression_predictions)
        print "Confusion-Matrix (with bins):"
        print confusionMatrix

        if save_cm:
            with open(save_cm, "wb") as fp:
                pickle.dump(confusionMatrix, fp)

        coord.request_stop()
        coord.join(threads)
        sess.close()

        return return_predictions, return_labels


if __name__ == '__main__': main()
