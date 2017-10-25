""" rates a single image (viewing angle) only by tumordistance """

import argparse

import rate_tumordistance

parser = argparse.ArgumentParser()
parser.add_argument('image_path')
parser.add_argument('--num_tumors', type=int, default=3)
args = parser.parse_args()

rate_tumordistance.rateImage(args.image_path, args.num_tumors, verbose=True, plot=True)

