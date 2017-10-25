import argparse
import os


def split(ratings_file_path, test_livers):
    """ splits a ratings-file in test- and train-file """

    ratings_file = open(ratings_file_path)
    ratings_train_path = os.path.join(os.path.dirname(ratings_file_path), "ratings_train.txt")
    ratings_test_path = os.path.join(os.path.dirname(ratings_file_path), "ratings_test.txt")
    ratings_train = open(ratings_train_path, mode='w')
    ratings_test = open(ratings_test_path, mode='w')

    for line in ratings_file:
        belongsToTest = False
        for liverpattern in test_livers:
            if liverpattern in line:
                ratings_test.writelines(line)
                belongsToTest = True
        if not belongsToTest:
            ratings_train.writelines(line)

    ratings_train.close()
    ratings_test.close()
    return ratings_train_path, ratings_test_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ratings_file', type=str)
    parser.add_argument('--test_livers', type=int, nargs='+', default=[])
    args = parser.parse_args()
    test_livers = ["gen_liver_{0}_".format(i) for i in args.test_livers]
    split(args.ratings_file, test_livers)

if __name__ == '__main__':
    main()
