""" prints a bin-distribution of a ground-truth-file (ratings.txt) """


import numpy as np
import argparse

def getBinFromRating(rating, num_bins):
    return int(np.clip(np.floor(rating * num_bins), 0, num_bins - 1))

def getBinDistribution(ratingsFilePath, num_bins):
    """ returns the distribution of files into the specified number of bins """
    bins = [[] for x in range(num_bins)]
    ratingsFile = open(ratingsFilePath)
    for line in ratingsFile:
        arrayLine = line.rstrip().split(" ")
        imagePath = arrayLine[0]
        rating = float(arrayLine[1])
        bin = getBinFromRating(rating , num_bins)
        bins[bin].append((imagePath, rating))
    return bins

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ratings_file_path', type=str)
    parser.add_argument('--num_bins', type=int, default=10)
    args = parser.parse_args()
    bins = getBinDistribution(args.ratings_file_path, args.num_bins)
    for i in range(args.num_bins):
        print "[{0}] {1}".format(i, len(bins[i]))

if __name__ == '__main__': main()