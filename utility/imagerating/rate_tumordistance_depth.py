""" rates images by tumordistance and distance from the tumors to the surface of the liver by using depth-values """

import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os
from trianglesolver import solve, degree
import Imath
import OpenEXR
import sys
import math
from PIL import Image


def getSmallestDepthInSurrounding(depth_image, x, y):
    kernel_x = np.linspace(-1, 1, 3, endpoint=True)
    kernel_y = np.linspace(-1, 1, 3, endpoint=True)
    min_dist = sys.float_info.max
    for dx in kernel_x:
        for dy in kernel_y:
            point_x = int(x + dx)
            point_y = int(y + dy)
            if 0 <= point_x <= depth_image.shape[1] and 0 <= point_y <= depth_image.shape[0]:
                dist = depth_image[point_y][point_x]
                if dist < min_dist:
                    min_dist = dist
    return min_dist


def exrToNumpy(exrImagePath):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    exrFile = OpenEXR.InputFile(exrImagePath)
    dw = exrFile.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    redstr = exrFile.channel('R', pt)
    red = np.fromstring(redstr, dtype=np.float32)
    red.shape = (size[1], size[0])  # Numpy arrays are (row, col)
    return red


def getDepthImage(image_path_exr):
    """ returns a PIL Image from a path to an OpenEXR-Image """
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    image_exr = OpenEXR.InputFile(image_path_exr)
    dw = image_exr.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    redstr = image_exr.channel('R', pt)  # channels should all contain depth
    image = Image.frombytes("F", size, redstr)
    return image


def rateImage(image_path, depth_tumors_path, depth_liver_path, num_tumors, verbose=False, plot=False, images=None):
    """ rates a given image by summing up tumor distances and comparing to enclosing circle of liver """
    if not images:
        img = cv2.imread(image_path)
        # get depth images
        depth_tumors = exrToNumpy(depth_tumors_path)
        depth_liver = exrToNumpy(depth_liver_path)
    else:
        img = images[0]
        depth_tumors = images[1]
        depth_liver = images[2]
    num_tumors = num_tumors
    b, g, r = cv2.split(img)

    # threholding, closing of tumors/liver
    ret, thresh1 = cv2.threshold(b, 75, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(r, 75, 277, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    dilation_kernel = np.ones((2,2), np.uint8)
    closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    closing_liver = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
    # dilation to make really small tumors bigger, because otherwise opencv will not recognize them, even if they are visible
    closing = cv2.dilate(closing, dilation_kernel, iterations=1)
    if plot:
        plt.subplot(151), plt.imshow(cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)), plt.title("closing")
    dst = cv2.addWeighted(img, 0.7, cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR), 0.3, 0.0)

    _, contours_tumors, _ = cv2.findContours(closing.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    _, contours_liver, _ = cv2.findContours(closing_liver.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(dst, contours_tumors, -1, (255, 0, 0), 1)

    # sum up liver contours and tumor area
    area_liver = cv2.contourArea(contours_liver[0])
    liverPts = contours_liver[0]
    for contour in contours_liver[1:]:
        area_liver += cv2.contourArea(contour)
        liverPts = np.concatenate((liverPts, contour))

    area_tumors = 0
    for contour in contours_tumors:
        area_tumors += cv2.contourArea(contour)

    # calculate tumor momentums -> centers
    centers = []
    for i in range(len(contours_tumors)):
        moments = cv2.moments(contours_tumors[i])
        # check m00 because division by zero ??? profit
        has_zero_moments = moments['m00'] == 0.0
        if verbose:
            print(moments['m00'])
        if not has_zero_moments:
            centers.append((moments['m10'] / moments['m00'], moments['m01'] / moments['m00']))
        else:
            pass
            # print "a tumor has zero momentum"

    # check if at least 2 tumors are visible -> else no distance possible!
    if len(centers) > 1:
        pts = np.array(centers, np.int32)
        pts = pts.reshape((-1, 1, 2))
        # calculate hull of tumor points
        hull = cv2.convexHull(pts)
        cv2.drawContours(dst, [hull], 0, (255, 255, 255))
        # calculate enclosing circle of liver
        (x, y), radius = cv2.minEnclosingCircle(liverPts)
        center = (int(x), int(y))
        # center = (int(y), int(x))
        radius = int(0.60 * radius)
        cv2.circle(dst, center, radius, (255, 255, 255), 1)
        # calculate distances of tumors (arclength)
        arcLength = cv2.arcLength(hull, True)
        # calculate max score possible distance
        angle = 360.0 / num_tumors * degree
        if verbose:
            print("calculating maxDistance: a={0}, b={1}, C={2}".format(radius, radius, angle))
        _, _, maxDistance, _, _, _ = solve(a=radius, b=radius, C=angle)
        score = arcLength / (maxDistance * num_tumors)
        # punish views with less tumors
        punish_factor = len(centers) / float(num_tumors) if len(centers) <= num_tumors else 1
        # score = score * punish_factor
        if score > 1:
            score = 1
        # grab depth values of centers
        dist_avg = 0
        for (x, y) in centers:
            x = int(x)
            y = int(y)
            dt = getSmallestDepthInSurrounding(depth_tumors, x, y)
            dl = getSmallestDepthInSurrounding(depth_liver, x, y)
            dist = dt - dl
            dist_avg += dist
            if verbose:
                print("distance tumor to liver at ({0}, {1}): {2}".format(x, y, dist))
                print(
                    "exact values liver: {0} tumors: {1}".format(depth_liver[y, x], depth_tumors[y, x]))
        dist_avg /= len(centers)
        if verbose:
            print("average tumor to liver distance:", dist_avg)
        # distance rating
        maxDistance = 100
        dist_margin = 10
        if dist_avg <= dist_margin:
            dist_score = 1
        else:
            # solve equation for score function
            a = np.array([[dist_margin, 1], [100, 1]])
            b = np.array([1, 0])
            solution = np.linalg.solve(a, b)
            m = solution[0]
            b = solution[1]
            dist_score = m * dist_avg + b
        # dist_score = 1.0 - ((dist_avg)/ maxDistance)
        if dist_score > 1:
            dist_score = 1
        elif dist_score < 0:
            dist_score = 0
    else:
        arcLength = 0
        score = 0
        dist_score = 0
        punish_factor = 0

    score_final = math.pow(0.5 * score + 0.5 * dist_score, 2) * punish_factor

    if verbose:
        # print "len(contours_liver)", len(contours_liver)
        print("#####\n\n")
        print("dist_score", dist_score, "score", score)
        print("final_score: ", score_final)
        print("arcLength: ", arcLength)
        print("Tumorezentren:", centers)
        print("sichtbare Tumore:", len(contours_tumors))
        print("Leberflaeche: ", area_liver)
        print("Tumorflaeche: ", area_tumors)
    if plot:
        plt.subplot(152), plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)), plt.title("dst")
        plt.subplot(153), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("img")
        dt = exrToNumpy(depth_tumors_path)
        dl = exrToNumpy(depth_liver_path)
        plt.subplot(154), plt.imshow(dt), plt.title("dt")
        plt.subplot(155), plt.imshow(dl), plt.title("dl")
        plt.show()

    return score_final, len(centers)


def rateImages(images_path, results_path, num_tumors=3, rotationsfile=None):
    results = open(results_path, 'w+')
    if not rotationsfile:
        for root, directories, filenames in os.walk(images_path):
            for file in filenames:
                if "liver" in file and os.path.splitext(file)[-1] == '.png':
                    relative_path = os.path.join(os.path.basename(root), file)
                    image_path = os.path.join(root, file)
                    basename = os.path.splitext(file)[0]
                    depth_tumors_path = os.path.join(root, basename + "_dt.exr")
                    depth_liver_path = os.path.join(root, basename + "_dl.exr")
                    score, visible_tumors = rateImage(image_path, depth_tumors_path, depth_liver_path, num_tumors)
                    results.write(relative_path + " " + str(score) + " " + str(visible_tumors) + " tumors" + "\n")
    # not dry and stupid, only works for bruteforce rendering (with 1 image)
    else:
        rotations = open(rotationsfile)
        for line in rotations:
            liverName = line.split()[0]
            image_path = os.path.join(images_path, liverName, "liver_0.png")
            depth_tumors_path = os.path.join(images_path, liverName, "liver_0_dt.exr")
            depth_liver_path = os.path.join(images_path, liverName, "liver_0_dl.exr")
            score, visible_tumors = rateImage(image_path, depth_tumors_path, depth_liver_path, num_tumors)
            results.write(os.path.join(liverName, "liver_0.png") + " " + str(score) + " " + str(visible_tumors) + " tumors" + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_path')
    parser.add_argument('results_path')
    parser.add_argument('--num_tumors', type=int, default=3)
    args = parser.parse_args()
    rateImages(args.images_path, args.results_path, args.num_tumors)


if __name__ == '__main__': main()
