""" starts the training of the rating networks """

import argparse
import ratings_reader
from networks import *

from tf_helper.train_mse_rating import MSERegressionTrainerRating
from tf_helper.train_crossentropy import CrossentropyClassificationTrainer


def parseArguments():
    parser = argparse.ArgumentParser(description='Train rating network')
    parser.add_argument('--num_epochs', default=25, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=10, type=int, help='batch size for training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--input_format', nargs='+', type=int, default=[1, 100, 100, 3],
                        help='input format: [num_images, width, height, depth]')
    parser.add_argument('--nntype', type=str, default='fc_rating', help="neural network to use")
    parser.add_argument('--output_classification', type=bool, default=False,
                        help="use classification instead of regression")
    parser.add_argument('--num_classes', type=int, default=100, help="number of classes if classification is used")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--no_norm_images', dest='norm_images', action='store_false', default=True)
    parser.add_argument('--no_norm_depth', dest='norm_depth', action='store_false', default=True)
    parser.add_argument('--augmentation', action='store_true', default=False)
    parser.add_argument('--accuracy_error', type=float, default=0.05)
    parser.add_argument('--no_depth', action='store_true', default=False)
    return parser


def main():
    parser = parseArguments()
    mainParser = argparse.ArgumentParser(add_help=False, parents=[parser])
    mainParser.add_argument('train_data', type=str, help='path to train data (tf-records file)')
    mainParser.add_argument('test_data', type=str, help='path to test data (tf-records file)')
    mainParser.add_argument('--num_train', default=1000, type=int, help='number of train samples')
    mainParser.add_argument('--num_test', default=100, type=int, help='number of test samples')
    mainParser.add_argument('--save_path', type=str, default=".", help='path where log and model are saved')
    args = mainParser.parse_args()
    train(args.train_data, args.test_data, args.num_train, args.num_test, args.save_path, args)

def train(train_data, test_data, num_train, num_test, save_path, args):
    try:
        if not args.output_classification:
            network_model = getattr(globals()[args.nntype], args.nntype)()
        else:
            network_model = getattr(globals()[args.nntype], args.nntype)(number_of_outputs=args.num_classes)
        network_model.dropout = args.dropout
    except KeyError:
        print "Network does not exist!"
        return
    num_images = args.input_format[0]
    image_width = args.input_format[1]
    image_height = args.input_format[2]
    image_depth = args.input_format[3]
    train_data = train_data
    test_data = test_data
    learning_rate = args.learning_rate

    print("train_data: " + train_data)
    print("test_data: " + test_data)

    # initialize readers
    reader_train = ratings_reader.RatingsReader(train_data, batch_size=args.batch_size, image_width=image_width,
                                                image_height=image_height, image_depth=image_depth,
                                                output_classification=args.output_classification,
                                                num_classes=args.num_classes, shuffle_batch=True,
                                                norm_images=args.norm_images, norm_depth=args.norm_depth,
                                                augmentation=args.augmentation, no_depth=args.no_depth)
    reader_test = ratings_reader.RatingsReader(test_data, batch_size=args.batch_size, image_width=image_width,
                                               image_height=image_height, image_depth=image_depth,
                                               output_classification=args.output_classification,
                                               num_classes=args.num_classes, shuffle_batch=True,
                                               norm_images=args.norm_images, norm_depth=args.norm_depth,
                                               augmentation=args.augmentation, no_depth=args.no_depth)
    if not args.output_classification:
        print "Training Regression-Network with MSE"
        networkTrainer = MSERegressionTrainerRating(reader_train, reader_test, args.num_epochs,
                                                    num_train, num_test, save_path,
                                                    network_model, learning_rate=learning_rate,
                                                    accuracy_error=args.accuracy_error, batch_size=args.batch_size)
    else:
        print "Training Classification-Network with Cross-Entropy"
        networkTrainer = CrossentropyClassificationTrainer(reader_train, reader_test, args.num_epochs,
                                                           num_train, num_test, save_path,
                                                           network_model, learning_rate=learning_rate)
    networkTrainer.train()


if __name__ == '__main__':
    main()
