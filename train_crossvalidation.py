""" Trains a network with cross-validation, can also be used to generate train/test pairs
    with a list (cv_file) of how to seperate train and test """

import argparse
import os
import shutil
from utility.datageneration.split import split
from utility.upscale_rating_data import upscale
from rating_network.convert import RatingRecordConverter
from rating_network.train import parseArguments, train


def getTestFolds(cv_file):
    file = open(cv_file)
    folds = [[]]
    foldnumber = 0
    for line in file:
        if line.strip() == '':
            folds.append([])
            foldnumber += 1
        else:
            arrayLine = line.rstrip().split(" ")
            folds[foldnumber].append(arrayLine[0])
    return folds


def train_fold(train_records_path, test_records_path, fold_save_path, num_train, num_test, args):
    train(train_records_path, test_records_path, num_train, num_test, fold_save_path, args)


def main():
    # get parser arguments for training rating network
    train_parser = parseArguments()
    # add parser arguments for crossvalidation
    parser = argparse.ArgumentParser(parents=[train_parser], add_help=False)
    parser.add_argument('ratings_file', type=str, help='ratings file of full dataset used for splitting')
    parser.add_argument('cv_file', type=str, help='file with a list of the folds of the test-set for crossvalidation')
    parser.add_argument('upscale_to', type=int, help='number of samples per bin in train dataset after augmentation')
    parser.add_argument('save_path', type=str, help='path to folder with will contain the trained models')
    parser.add_argument('--num_bins', type=int, default=10, help='number of bins used for upscaling train data')
    parser.add_argument('--num_tumors', type=int, default=3, help='number of tumors used for rating in upscaling')
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--remove_train_data', action='store_true', default=False)
    parser.add_argument('--remove_test_data', action='store_true', default=False)
    parser.add_argument('--only_gen_data', action='store_true', default=False)
    args = parser.parse_args()
    test_liver_folds = getTestFolds(args.cv_file)
    if args.verbose:
        for i, fold in enumerate(test_liver_folds):
            print "fold " + str(i)
            print fold

    # iterate through folds
    for foldnumber, fold in enumerate(test_liver_folds):
        # split fold, generate ratings_train.txt and ratings_test.txt
        ratings_train_path, ratings_test_path = split(args.ratings_file, fold)

        # upscale training data
        data_path = os.path.dirname(args.ratings_file)
        augmented_ratings_train_path, augmented_data_path = upscale(data_path, args.upscale_to, args.num_bins,
                                                                    args.num_tumors, ratings_train_path,
                                                                    "cv_train_fold{0}".format(foldnumber),
                                                                    force_delete=True)

        # convert train and test to tfrecords
        train_converter = RatingRecordConverter(data_path, augmented_ratings_train_path, shuffle=True,
                                                name="train_fold{0}.tfrecords".format(foldnumber))
        train_converter.convert()
        test_converter = RatingRecordConverter(data_path, ratings_test_path,
                                               name="test_fold{0}.tfrecords".format(foldnumber))
        test_converter.convert()

        # train
        num_train = sum(1 for line in open(augmented_ratings_train_path))
        num_test = sum(1 for line in open(ratings_test_path))
        train_records_path = os.path.join(data_path, "train_fold{0}.tfrecords".format(foldnumber))
        test_records_path = os.path.join(data_path, "test_fold{0}.tfrecords".format(foldnumber))
        if not args.only_gen_data:
            save_path = os.path.join(args.save_path, "fold_{0}".format(foldnumber))
            train_fold(train_records_path, test_records_path, save_path, num_train, num_test, args)

        # delete tfrecords and augmented data
        if args.remove_train_data:
            os.remove(train_records_path)
            os.remove(augmented_ratings_train_path)
            os.remove(ratings_train_path)
            shutil.rmtree(augmented_data_path)
        if args.remove_test_data:
            os.remove(test_records_path)
            os.remove(ratings_test_path)


if __name__ == '__main__':
    main()
