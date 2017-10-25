from itertools import izip
from PIL import Image
import os


def compareImages(image1, image2):
    i1 = Image.open(image1)
    i2 = Image.open(image2)
    assert i1.mode == i2.mode, "Different kinds of images."
    assert i1.size == i2.size, "Different sizes."

    pairs = izip(i1.getdata(), i2.getdata())
    if len(i1.getbands()) == 1:
        # for gray-scale jpegs
        dif = sum(abs(p1 - p2) for p1, p2 in pairs)
    else:
        dif = sum(abs(c1 - c2) for p1, p2 in pairs for c1, c2 in zip(p1, p2))

    ncomponents = i1.size[0] * i1.size[1] * 3
    if dif < 10000:
        print "same images {0} {1}".format(image1, image2)


def main():
    dir = os.listdir(".")
    liverDirs = [f for f in dir if "liver" in f]
    # for dir1 in liverDirs:
    #     images = os.listdir(dir1)
    allImages = []
    for root, directories, filenames in os.walk("."):
        for file in filenames:
            if "liver" in file:
                allImages.append(root + "/" + file)

    for image1 in allImages:
        print "comparing {0}".format(image1)
        for image2 in allImages:
            if image1 != image2:
                compareImages(image1, image2)


if __name__ == '__main__': main()
