""" compares images of datasets with hash-function """

from itertools import izip
from PIL import Image
import os
import imagehash
import sys
import argparse


def parseArguments():
    parser = argparse.ArgumentParser(description='Compare Images of two Datasets')
    parser.add_argument('results_file1', type=str,
                        help="first ratings file (train-data if duplicates should be removed)")
    parser.add_argument('results_file2', type=str,
                        help="second ratings file (test-data if duplicates should be removed)")
    parser.add_argument('--new_ratings', type=str, default=None)
    return parser.parse_args()


def main():
    def containsTrainAndTest(dataset_one, dataset_two, imagelist):
        contains_test = False
        contains_train = False
        for item in imagelist:
            if dataset_one in item:
                contains_train = True
            if dataset_two in item:
                contains_test = True
        return (contains_train and contains_test)

    def addImageHash(hashdict, image):
        hash = imagehash.dhash(Image.open(image))
        if hash in hashdict.keys():
            hashdict[hash].append(image)
        else:
            hashdict[hash] = [image]
        return hash

    def printSimilarImages(dataset_one, dataset_two, hashdict):
        similar = 0
        similarList = []
        for hash, imageList in hashdict.iteritems():
            if containsTrainAndTest(dataset_one, dataset_two, imageList):
                print imageList
                similar += len(imageList)
                similarList += imageList
        print similar, "similar Images\n\n"
        return similarList

    def getImageList(datasetPath, ratingsFile):
        images = []
        for line in ratingsFile:
            arrayLine = line.rstrip().split(" ")
            image = os.path.join(datasetPath, arrayLine[0])
            if os.path.isdir(image):
                images += [os.path.join(image, f) for f in os.listdir(image) if f.endswith(".png")]
            else:
                images.append(image)
        return images

    args = parseArguments()

    hashes = {}

    ratings_one = open(args.results_file1)
    ratings_two = open(args.results_file2)

    dataset_one = os.path.dirname(os.path.realpath(args.results_file1))
    dataset_two = os.path.dirname(os.path.realpath(args.results_file2))

    images = getImageList(dataset_one, ratings_one)
    images += getImageList(dataset_two, ratings_two)

    for image in images:
        addImageHash(hashes, image)

    similarList = printSimilarImages(dataset_one, dataset_two, hashes)

    if args.new_ratings:
        oldRatingsFile = open(args.results_file1)
        newRatingsFile = open(args.new_ratings, 'w+')
        for line in oldRatingsFile:
            arrayLine = line.rstrip().split(" ")
            image = os.path.join(dataset_one, arrayLine[0])
            if image not in similarList:
                newRatingsFile.write(line)
    oldRatingsFile.close()
    newRatingsFile.close()


if __name__ == '__main__': main()
