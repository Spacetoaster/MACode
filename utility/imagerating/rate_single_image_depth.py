""" rates a single image using tumordistance and depth """

import argparse
import os
import rate_tumordistance_depth

parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str)
parser.add_argument('--num_tumors', type=int, default=3)
args = parser.parse_args()

root = os.path.dirname(args.image_path)
basename = os.path.splitext(os.path.basename(args.image_path))[0]
depth_tumors_path = os.path.join(root, basename + "_dt.exr")
depth_liver_path = os.path.join(root, basename + "_dl.exr")

rate_tumordistance_depth.rateImage(args.image_path, depth_tumors_path, depth_liver_path, args.num_tumors,
                                   verbose=True, plot=True)
