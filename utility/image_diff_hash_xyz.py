from itertools import izip
from PIL import Image
import os
import imagehash


def main():
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
            if len(imageList) > 1:
                print imageList
                similar += len(imageList)
        print similar, "similar Images\n\n"

    dir = os.listdir(".")


    hashes_x = {}
    hashes_y = {}
    hashes_z = {}
    hashes_xyz = {}

    imageDirs = filter(os.path.isdir, os.listdir("."))
    for dir in imageDirs:
        hash_x = addImageHash(hashes_x, dir + "/liver_x.png")
        hash_y = addImageHash(hashes_y, dir + "/liver_y.png")
        hash_z = addImageHash(hashes_z, dir + "/liver_z.png")
        hash_xyz = str(hash_x) + str(hash_y) + str(hash_z)
        if hash_xyz in hashes_xyz.keys():
            hashes_xyz[hash_xyz].append(dir)
        else:
            hashes_xyz[hash_xyz] = [dir]


    # allImages = []
    # for root, directories, filenames in os.walk("."):
    #     for file in filenames:
    #         if "liver" in file:
    #             allImages.append(root + "/" + file)
    #
    # for image in allImages:
    #     if "liver_x" in image:
    #         addImageHash(hashes_x, image)
    #     if "liver_y" in image:
    #         addImageHash(hashes_y, image)
    #     if "liver_z" in image:
    #         addImageHash(hashes_z, image)

    print "x images:"
    printSimilarImages(hashes_x)
    print "y images:"
    printSimilarImages(hashes_y)
    print "z images:"
    printSimilarImages(hashes_z)
    print "xyz images:"
    printSimilarImages(hashes_xyz)




if __name__ == '__main__': main()
