from itertools import izip
from PIL import Image
import os
import imagehash
import sys


def main():
    def containsTrainAndTest(imagelist):
        contains_test = False
        contains_train = False
        for item in imagelist:
            if dir_train in item:
                contains_train = True
            if dir_test in item:
                contains_test = True
        return (contains_train and contains_test)

    def addImageHash(hashdict, image):
        hash = imagehash.dhash(Image.open(image))
        if hash in hashdict.keys():
            hashdict[hash].append(image)
        else:
            hashdict[hash] = [image]
        return hash

    def printSimilarImages(hashdict):
        similar = 0
        for hash, imageList in hashdict.iteritems():
            if containsTrainAndTest(imageList):
                print imageList
                similar += len(imageList)
        print similar, "similar Images\n\n"

    dir_train = str(sys.argv[1])
    dir_test = str(sys.argv[2])

    hashes_x = {}
    hashes_y = {}
    hashes_z = {}
    hashes_xyz = {}

    imageDirs_train = [os.path.join(dir_train, f) for f in os.listdir(dir_train) if
                       os.path.isdir(os.path.join(dir_train, f))]
    imageDirs_test = [os.path.join(dir_test, f) for f in os.listdir(dir_test) if
                      os.path.isdir(os.path.join(dir_test, f))]

    imageDirs = imageDirs_train + imageDirs_test
    for dir in imageDirs:
        hash_x = addImageHash(hashes_x, dir + "/liver_x.png")
        hash_y = addImageHash(hashes_y, dir + "/liver_y.png")
        hash_z = addImageHash(hashes_z, dir + "/liver_z.png")
        hash_xyz = str(hash_x) + str(hash_y) + str(hash_z)
        if hash_xyz in hashes_xyz.keys():
            hashes_xyz[hash_xyz].append(dir)
        else:
            hashes_xyz[hash_xyz] = [dir]

    print "x images:"
    printSimilarImages(hashes_x)
    print "y images:"
    printSimilarImages(hashes_y)
    print "z images:"
    printSimilarImages(hashes_z)
    print "xyz images:"
    printSimilarImages(hashes_xyz)


if __name__ == '__main__': main()
