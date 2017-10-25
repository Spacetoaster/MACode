""" evaluates results from network-based methods (rotation network, combination) """

import rotation_network.predict
import subprocess
import argparse
import os
from utility.imagerating.rate_tumordistance_depth import rateImage
import numpy as np

parser = argparse.ArgumentParser(description='Evaluate Network')
parser.add_argument('predict_data', type=str, help='path to test data (tf-records file)')
parser.add_argument('--num_samples', default=100, type=int, help='number of samples for prediction')
parser.add_argument('--input_format', nargs='+', type=int, default=[3, 100, 100, 3],
                    help='input format: [num_images, width, height, depth]')
parser.add_argument('--nntype', type=str, default='someNetwork', help="neural network to use")
parser.add_argument('--blender_path', type=str, default="/Applications/blender.app/Contents/MacOS/blender")
parser.add_argument('--render_script_path', type=str,
                    default="/Users/Spacetoaster/Desktop/scripts/utility/datageneration/render.py")
parser.add_argument('--render_path', type=str, default=None, help='path to folder to use for rendering')
parser.add_argument('--liver_path', type=str, default=None,
                    help='path to livers to use for prediction/dataset generation')
args = parser.parse_args()

blenderPath = args.blender_path
renderPath = args.render_script_path
renderedPath = args.render_path
liverPath = args.liver_path
predictions, labels, names = rotation_network.predict.predict(args.predict_data, args.num_samples, args.input_format,
                                                              args.nntype, withName=True)

deviations = []

for p, l, n in zip(predictions, labels, names):
    # render prediction
    print str(p[0]), str(p[1]), str(p[2]), str(p[3]), str(0.0)
    subprocess.call([blenderPath, "--background", "--python", renderPath, "--", "--blender_files", liverPath,
                     "--rendered_dir", renderedPath, "--render_depth", "--only_rotation",
                     "{0:.5f}".format(p[0]), "{0:.5f}".format(p[1]), "{0:.5f}".format(p[2]), "{0:.5f}".format(p[3]),
                     "{0:.1f}".format(0.0), "--num_rotations=2", "--only_liver", str(n)])

    # render label
    print str(l[0]), str(l[1]), str(l[2]), str(l[3]), str(0.0)
    subprocess.call([blenderPath, "--background", "--python", renderPath, "--", "--blender_files", liverPath,
                     "--rendered_dir", renderedPath, "--render_depth", "--only_rotation",
                     "{0:.5f}".format(l[0]), "{0:.5f}".format(l[1]), "{0:.5f}".format(l[2]), "{0:.5f}".format(l[3]),
                     "{0:.1f}".format(1.0), "--num_rotations=2", "--only_liver", str(n)])
    # rate images
    predictedImagePath = os.path.join(renderedPath, n + "_0.0")
    labelImagePath = os.path.join(renderedPath, n + "_1.0")
    predictedRating = rateImage(os.path.join(predictedImagePath, "liver_0.png"), os.path.join(predictedImagePath, "liver_0_dt.exr"),
                                os.path.join(predictedImagePath, "liver_0_dl.exr"), 3)
    labelRating = rateImage(os.path.join(labelImagePath, "liver_0.png"), os.path.join(labelImagePath, "liver_0_dt.exr"),
                                os.path.join(labelImagePath, "liver_0_dl.exr"), 3)

    deviations.append(abs(predictedRating[0] - labelRating[0]))

mean = np.mean(deviations)
print "avg rating deviation: ", mean