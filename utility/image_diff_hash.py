from itertools import izip
from PIL import Image
import os
import imagehash
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratings_file', type=str, default=None)
    args = parser.parse_args()
    dir = os.listdir(".")

    allImages = []
    if args.ratings_file:
        ratings = open(args.ratings_file)
        for line in ratings:
            arrayLine = line.rstrip().split(" ")
            allImages.append(arrayLine[0])
    else:
        for root, directories, filenames in os.walk("."):
            for file in filenames:
                if "liver" in file and "png" in file:
                    allImages.append(root + "/" + file)

    hashes = {}

    for image in allImages:
        hash = imagehash.dhash(Image.open(image))
        if hash in hashes.keys():
            hashes[hash].append(image)
        else:
            hashes[hash] = [image]

    similar = 0
    for hash, imageList in hashes.iteritems():
        if len(imageList) > 1:
            print imageList
            similar += len(imageList)
    print similar, "similar Images"

if __name__ == '__main__': main()
