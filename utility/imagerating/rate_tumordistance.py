""" rates images by tumordistance """

import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import sys
import os
from trianglesolver import solve, degree


def rateImage(image_path, num_tumors, verbose=False, plot=False):
    """ rates a given image by summing up tumor distances and comparing to enclosing circle of liver """
    img = cv2.imread(image_path)
    num_tumors = num_tumors
    b, g, r = cv2.split(img)

    # threholding, closing of tumors/liver
    ret, thresh1 = cv2.threshold(b, 75, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(r, 75, 277, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    closing_liver = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
    if plot:
        plt.subplot(131), plt.imshow(cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)), plt.title("closing")
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
            print moments['m00']
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
        radius = int(0.65 * radius)
        cv2.circle(dst, center, radius, (255, 255, 255), 1)
        # calculate distances of tumors (arclength)
        arcLength = cv2.arcLength(hull, True)
        # calculate max score possible distance
        angle = 360.0 / num_tumors * degree
        if verbose:
            print "calculating maxDistance: a={0}, b={1}, C={2}".format(radius, radius, angle)
        _, _, maxDistance, _, _, _ = solve(a=radius, b=radius, C=angle)
        score = arcLength / (maxDistance * num_tumors)
        # punish views with less tumors
        punish_factor = len(centers) / float(num_tumors) if len(centers) <= num_tumors else 1
        score = score * punish_factor
        if score > 1:
            score = 1
    else:
        score = 0

    if verbose:
        print "len(contours_liver)", len(contours_liver)
        print "Score: ", score
        print "arcLength: ", arcLength
        print "Tumorezentren:", centers
        print "sichtbare Tumore:", len(contours_tumors)
        print "Leberflaeche: ", area_liver
        print "Tumorflaeche: ", area_tumors
    if plot:
        plt.subplot(132), plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)), plt.title("dst")
        plt.subplot(133), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("img")
        plt.show()

    return score, len(centers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_path')
    parser.add_argument('results_path')
    parser.add_argument('--num_tumors', type=int, default=3)
    args = parser.parse_args()

    results = open(args.results_path, 'w+')
    for root, directories, filenames in os.walk(args.images_path):
        for file in filenames:
            if "liver" in file:
                image_path = os.path.join(root, file)
                score, visible_tumors = rateImage(image_path, args.num_tumors)
                results.write(image_path + " " + str(score) + " " + str(visible_tumors) + " tumors" + "\n")


if __name__ == '__main__': main()
