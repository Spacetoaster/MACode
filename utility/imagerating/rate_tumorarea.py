""" rates an image by tumor-area """

import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import sys
import os
import math
from trianglesolver import solve, degree

def rateImage(image_path, num_tumors, verbose=False, plot=False):
    """ rates a given image by summing up tumor distances and comparing to enclosing circle of liver """
    img = cv2.imread(image_path)
    num_tumors = num_tumors
    b,g,r = cv2.split(img)
    
    # threholding, closing of tumors/liver
    ret, thresh1 = cv2.threshold(b,75,255,cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(r,75,277,cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    closing_liver = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
    if plot:
        plt.subplot(131), plt.imshow(cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)), plt.title("closing")
    dst = cv2.addWeighted(img, 0.7, cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR), 0.3, 0.0)
    
    _, contours_tumors, _ = cv2.findContours(closing.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    _, contours_liver, _ = cv2.findContours(closing_liver.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(dst, contours_tumors, -1, (255,0,0),1)
    
    # sum up liver contours and tumor area
    area_liver = cv2.contourArea(contours_liver[0])
    liverPts = contours_liver[0]
    for contour in contours_liver[1:]:
        area_liver += cv2.contourArea(contour)
        liverPts = np.concatenate((liverPts, contour))
    
    area_tumors = 0
    for contour in contours_tumors:
        area_tumors += cv2.contourArea(contour)

    score = area_tumors / area_liver

    if verbose:
        print "Score: ", score
        print "Leberflaeche: ", area_liver
        print "Tumorflaeche: ", area_tumors
    if plot:
        plt.subplot(132), plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)), plt.title("dst")
        plt.subplot(133), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("img")
        plt.show()

    return score


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
                score, visible_tumors  = rateImage(image_path, args.num_tumors)
                results.write(image_path + " " + str(score) + " " + str(visible_tumors) + " tumors" + "\n")

if __name__ == '__main__': main()
