""" prediction with a trained rating-model (classification) on a tfrecords-file """

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
    parser.add_argument('--num_classes', type=int, default=100, help="number of classes")
    parser.add_argument('--save_cm', type=str, default=None)
    return parser.parse_args()


def main():
    args = parseArguments()
    records_file = args.predict_data
    save_cm = args.save_cm

    try:
        network_model = getattr(globals()[args.nntype], args.nntype)(number_of_outputs=args.num_classes)
    except KeyError:
        print "Network does not exist!"
        return
    avg_loss = 0
    num_samples = args.num_samples

    graph = network_model
    test_reader = ratings_reader.RatingsReader(args.predict_data, batch_size=1, output_classification=True,
                                               num_classes=args.num_classes)
    network_model = Network(test_reader, test_reader, graph)

    print("Predicting [num_samples={0}]".format(num_samples))
    print("predict_data: " + records_file)

    net = network_model.output()
    prediction = tf.nn.softmax(net)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=network_model.label))

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(network_model.label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver()

    def predict_sample():
        inputs, label = network_model.getTrainBatch(sess)
        feed_dict = network_model.getFeedDict(inputs, label)
        prediction_result, prediction_loss = sess.run([prediction, loss], feed_dict)
        accu = tf.cast(tf.reduce_mean(accuracy), tf.float32).eval(feed_dict)
        return prediction_result, label, prediction_loss, accu

    with tf.Session() as sess:
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

        distance_one_acc = 0
        accuracys = []
        predictions = []
        labels = []
        for i in range(num_samples):
            result, label, my_loss, accu = predict_sample()
            # quaternion calculations
            avg_loss += my_loss
            result_class = tf.argmax(result, 1).eval()
            predictions.append(result_class)
            label_class = tf.argmax(label, 1).eval()
            labels.append(label_class)
            accuracys.append(accu)
            if abs(label_class[0] - result_class[0]) <= 1:
                distance_one_acc += 1
            print("result: {0} ({1})\tlabel: {2} \tloss: {3:.6f}".format(result_class, result, label_class, my_loss))
        avg_accuracy = tf.cast(tf.reduce_mean(accuracys), tf.float32).eval()
        avg_loss /= num_samples
        distance_one_acc /= float(num_samples)
        print("avg_loss: {0:.6f} \taccuracy: {1:.2f}".format(avg_loss, avg_accuracy))
        print("accuracy (with neighbors): {0:.2f}".format(distance_one_acc))
        confusionMatrix = sklearn.metrics.confusion_matrix(labels, predictions)
        print "Confusion-Matrix:"
        print confusionMatrix

        if save_cm:
            with open(save_cm, "wb") as fp:
                pickle.dump(confusionMatrix, fp)

        coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == '__main__': main()
