""" starts the training of a rotation network """

import argparse
import os

import tensorflow as tf

import quaternion_reader
from networks import *
from tf_helper.train_mse_rotation import MSERegressionTrainerRotation


def parseArguments():
    parser = argparse.ArgumentParser(description='Train rotation network')
    parser.add_argument('train_data', type=str, help='path to train data (tf-records file)')
    parser.add_argument('test_data', type=str, help='path to test data (tf-records file)')
    parser.add_argument('--num_epochs', default=25, type=int, help='number of epochs')
    parser.add_argument('--num_train', default=1000, type=int, help='number of train samples')
    parser.add_argument('--num_test', default=100, type=int, help='number of test samples')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--save_path', type=str, default=".", help='path where log and model are saved')
    parser.add_argument('--input_format', nargs='+', type=int, default=[3, 100, 100, 3],
                        help='input format: [num_images, width, height, depth]')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--nntype', type=str, default='someNetwork', help="neural network to use")
    parser.add_argument('--fc_multiplicator', type=int, default=1, help="multiplys number of fc units")
    parser.add_argument('--conv_multiplicator', type=int, default=1, help="multiplys number of fc units")
    parser.add_argument('--dropout', type=float, default=0.9, help="dropout")
    parser.add_argument('--error', type=int, default=25)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--combination', dest='combination', action='store_true', default=False)
    return parser.parse_args()


def main():
    args = parseArguments()
    try:
        network_model = getattr(globals()[args.nntype], args.nntype)(args.fc_multiplicator, args.conv_multiplicator,
                                                                     args.dropout)
    except KeyError:
        print "Network does not exist!"
        return
    num_images = args.input_format[0]
    image_width = args.input_format[1]
    image_height = args.input_format[2]
    image_depth = args.input_format[3]
    train_data = args.train_data
    test_data = args.test_data
    learning_rate = args.learning_rate

    print("train_data: " + train_data)
    print("test_data: " + test_data)

    # initialize readers
    liverReader_train = quaternion_reader.QuaternionReader(train_data, batch_size=args.batch_size,
                                                           image_width=image_width,
                                                           image_height=image_height,
                                                           image_depth=image_depth, number_of_images=num_images)
    liverReader_test = quaternion_reader.QuaternionReader(test_data, batch_size=args.batch_size,
                                                          image_width=image_width,
                                                          image_height=image_height,
                                                          image_depth=image_depth, number_of_images=num_images)
    networkTrainer = MSERegressionTrainerRotation(liverReader_train, liverReader_test, args.num_epochs, args.num_train,
                                                  args.num_test,
                                                  args.save_path, network_model, learning_rate=learning_rate,
                                                  batch_size=args.batch_size, accuracy_error=args.error,
                                                  pretrained=args.pretrained, combination=args.combination)
    networkTrainer.train()


if __name__ == '__main__': main()
