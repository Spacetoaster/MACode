""" scales rating-dataset so that bins are evenly distributed """

import cv2
import OpenEXR
import argparse
import numpy as np
import os
import Imath
import shutil
from imagerating import rate_tumordistance_depth
from datageneration.bin_distribution import getBinDistribution
from datageneration.bin_distribution import getBinFromRating



def rotate_image(rgb, dt, dl, angle):
    """ returns versions of the specified images rotated by the given angle"""
    width = rgb.shape[1]
    height = rgb.shape[0]
    pivot = (width / 2, height / 2)
    rotationMatrix = cv2.getRotationMatrix2D(pivot, angle, 1.0)
    bgr_rotated = cv2.warpAffine(rgb, rotationMatrix, (100, 100), borderMode=cv2.BORDER_REPLICATE)
    # no interpolation for depth values because that doesn't make a lot of sense and leads to values which are not
    # seen in the non-upscaled data
    dt_rotated = cv2.warpAffine(dt, rotationMatrix, (100, 100), flags=cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS,
                                borderMode=cv2.BORDER_REPLICATE)
    dl_rotated = cv2.warpAffine(dl, rotationMatrix, (100, 100), flags=cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS,
                                borderMode=cv2.BORDER_REPLICATE)
    return bgr_rotated, dt_rotated, dl_rotated


def save_exr_from_numpy(filename, image_as_array):
    """ saves an exr image (depth image) form a numpy array"""
    width = image_as_array.shape[1]
    height = image_as_array.shape[0]
    data = image_as_array.tostring()
    exr = OpenEXR.OutputFile(filename, OpenEXR.Header(width, height))
    exr.writePixels({'R': data, 'G': data, 'B': data})
    exr.close()


def exrToNumpy(exrImagePath):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    exrFile = OpenEXR.InputFile(exrImagePath)
    dw = exrFile.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    redstr = exrFile.channel('R', pt)
    red = np.fromstring(redstr, dtype=np.float32)
    red.shape = (size[1], size[0])  # Numpy arrays are (row, col)
    return red


def augmentImageByRotation(imagePath, numRotations, originalBin, data_path):
    """ returns a list of rotated versions of the passed image and creates the images/ratings in the augmented dir """
    angles = np.linspace(0, 360, numRotations + 1, endpoint=False)[1:]
    augmentedImages = []
    rgb = cv2.imread(os.path.join(data_path, imagePath))
    dt = exrToNumpy(os.path.join(os.path.dirname(os.path.join(data_path, imagePath)), 'liver_0_dt.exr'))
    dl = exrToNumpy(os.path.join(os.path.dirname(os.path.join(data_path, imagePath)), 'liver_0_dl.exr'))
    newRatings = open(new_ratings_file_path, 'a')
    generated_images = 0
    for i, angle in enumerate(angles):
        # try different offsets if exact rotation does not give the same bin as the original image
        offsets = np.linspace(0, 10, 100, endpoint=False)
        newBin = None
        save_version = False
        for offset in offsets:
            rgb_r, dt_r, dl_r = rotate_image(rgb, dt, dl, angle + offset)
            # rate image
            rating, _ = rate_tumordistance_depth.rateImage(None, None, None, num_tumors, images=[rgb_r, dt_r, dl_r])
            newBin = getBinFromRating(rating, num_bins)
            # if bins match, save image
            if originalBin == newBin:
                save_version = True
                break
        if save_version:
            rotDir = os.path.join(augmentedDataPath, os.path.dirname(imagePath) + "_rot" + str(i))
            os.makedirs(rotDir)
            # save images to rotDir
            rgb_path = os.path.join(rotDir, 'liver_0.png')
            dt_path = os.path.join(rotDir, 'liver_0_dt.exr')
            dl_path = os.path.join(rotDir, 'liver_0_dl.exr')
            cv2.imwrite(rgb_path, rgb_r)
            save_exr_from_numpy(dt_path, dt_r)
            save_exr_from_numpy(dl_path, dl_r)
            # make entry in new ratings file
            save_path = os.path.relpath(rgb_path, data_path)
            newRatings.write(getRatingsLine(save_path, rating))
            generated_images += 1
    newRatings.close()
    if generated_images == 0:
        print "Could not match bins. (" + imagePath + ")"
    return generated_images


def getRatingsLine(path, rating):
    return path + " " + str(rating) + "\n"


def reduceBin(bin, size, binLabel):
    """ copys a random suffled subset of the images in the bin to the new ratings file """
    print("reducing bin [" + str(binLabel) + "] (size: " + str(len(bin)) + ")")
    np.random.shuffle(bin)
    chosenImages = bin[:size]
    newRatings = open(new_ratings_file_path, 'a')
    for image in chosenImages:
        newRatings.write(getRatingsLine(image[0], image[1]))
    newRatings.close()


def augmentBin(bin, size, binLabel, data_path):
    """ copys the original images of the bin plus a random subset of synthesized images to reach the specified size """
    # copy ratings of the original images to the new ratings file
    newRatings = open(new_ratings_file_path, 'a')
    for imagePath, rating in bin:
        newRatings.write(getRatingsLine(imagePath, rating))
    newRatings.close()
    # determine number of left images and generate them
    augmentationFactor = np.ceil(float(size) / len(bin))
    print("augmenting bin [" + str(binLabel) + "] (size: " + str(len(bin)) + ", augmentationFactor: " + str(
        augmentationFactor) + ")")
    if augmentationFactor <= 1:
        return
    leftImages = size - len(bin)
    augmentedBin = []
    for imagePath, rating in bin:
        # determine how many images should be generated
        num_to_generate = augmentationFactor - 1
        actual_to_generate = num_to_generate if num_to_generate <= leftImages else leftImages
        num_generated = augmentImageByRotation(imagePath, actual_to_generate, binLabel, data_path)
        leftImages -= num_generated
        # break if no more images needed
        if leftImages <= 0:
            break


def removeExistingData(force=False):
    input = 'y'
    if not force:
        prompt = "augmented folder already exists. Remove existing augmented data? (y/n)\n"
        input = raw_input(prompt)
    if input == 'y':
        shutil.rmtree(augmentedDataPath)
        if os.path.exists(new_ratings_file_path):
            os.remove(new_ratings_file_path)
        return True
    else:
        return False


augmentedDataPath = None
new_ratings_file_path = None
num_tumors = None
num_bins = None

def upscale(data_path, upscale_to, number_of_bins, number_of_tumors, ratings_file, name, force_delete=False):
    global augmentedDataPath
    global new_ratings_file_path
    global num_tumors
    global num_bins
    num_bins = number_of_bins
    num_tumors = number_of_tumors
    if not ratings_file:
        ratings_file_path = os.path.join(data_path, "ratings.txt")
    else:
        ratings_file_path = ratings_file
    if not name:
        augmentedDataPath = os.path.join(data_path, "augmented")
        new_ratings_file_path = os.path.join(data_path, "augmented_ratings.txt")
    else:
        augmentedDataPath = os.path.join(data_path, "augmented_{0}".format(name))
        new_ratings_file_path = os.path.join(data_path, "augmented_ratings_{0}.txt".format(name))

    directory_clear = True
    if os.path.exists(augmentedDataPath):
        directory_clear = removeExistingData(force=force_delete)

    if directory_clear:
        if not os.path.exists(augmentedDataPath):
            os.makedirs(augmentedDataPath)
        bins = getBinDistribution(ratings_file_path, num_bins)
        for i, bin in enumerate(bins):
            if len(bin) == 0:
                print "Found bin with size 0, cannot augment data! aborting."
                removeExistingData(force=True)
                return
            if len(bin) > upscale_to:
                reduceBin(bin, upscale_to, i)
            elif len(bin) < upscale_to:
                augmentBin(bin, upscale_to, i, data_path)
    else:
        print "no new data generated"
    return new_ratings_file_path, augmentedDataPath

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('upscale_to', type=int)
    parser.add_argument('--num_bins', type=int, default=10)
    parser.add_argument('--num_tumors', type=int, default=3)
    parser.add_argument('--ratings_file', type=str, default=None)
    parser.add_argument('--name', type=str, default=None,
                        help="foldername will be augmented_{name} and filename augmented_ratings_{name}.txt")
    args = parser.parse_args()
    upscale(args.data_path, args.upscale_to, args.num_bins, args.num_tumors, args.ratings_file, args.name)


if __name__ == '__main__':
    main()
