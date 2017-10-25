""" sampling method (bruteforce), can also be used to generate ground-truth (optimal rotation) for a dataset """

import subprocess
from utility.imagerating.rate_tumordistance_depth import rateImages
import os, fnmatch
from rating_network import predict_regression
from rating_network.convert import RatingRecordConverter
import argparse
from pyquaternion import Quaternion
import math
import numpy as np
import tensorflow as tf
import pickle


def quaternionFromTuple(tuple):
    qw = float(tuple[0])
    qx = float(tuple[1])
    qy = float(tuple[2])
    qz = float(tuple[3])
    return Quaternion(qw, qx, qy, qz)


def correctPredictions(angleDiffs, axisDiffs, error):
    num_correct = 0
    for angleDiff, axisDiff in zip(angleDiffs, axisDiffs):
        if angleDiff <= error and axis_diff <= error:
            num_correct += 1
    return num_correct

parser = argparse.ArgumentParser(description='Bruteforce')
parser.add_argument('--liver_path', type=str, default=None, help='path to livers to use for prediction/dataset generation')
parser.add_argument('--render_path', type=str, default=None, help='path to folder to use for rendering')
parser.add_argument('--generate', type=str, default=None, help='path to results file in case of data generation')
parser.add_argument('--blender_path', type=str, default="/Applications/blender.app/Contents/MacOS/blender")
parser.add_argument('--render_script_path', type=str, default="/Users/Spacetoaster/Desktop/scripts/utility/datageneration/render.py")
parser.add_argument('--model_path', type=str, default="./model/cnn_reg/model")
parser.add_argument('--nntype', type=str, default="cnn_rating_regression")
parser.add_argument('--save_file', type=str, default="save_bruteforce")
parser.add_argument('--ico', type=int, default=3)
parser.add_argument('--randomRotation', type=int, default=None)
args = parser.parse_args()

blenderPath = args.blender_path
renderPath = args.render_script_path
liverPath = args.liver_path
renderedPath = args.render_path
ico_resolution = args.ico
if ico_resolution == 2:
    num_examples = 42
elif ico_resolution == 3:
    num_examples = 162
elif ico_resolution == 4:
    num_examples = 642
# num_examples = 162
if args.randomRotation:
    num_examples = args.randomRotation
num_tumors = 3
nntype = args.nntype
if args.generate:
    ratings_view = open(args.generate, 'w+')

liverFiles = fnmatch.filter(os.listdir(liverPath), "*.blend")

angleDiffs = []
axisDiffs = []
top10Predictions = []

allSortedPredictions = []
allSortedLabels = []

for liver in liverFiles:
    # render using blender
    if not args.randomRotation:
        subprocess.call([blenderPath, "--background", "--python", renderPath, "--", "--blender_files", liverPath,
                     "--rendered_dir", renderedPath, "--render_bruteforce_ico", str(ico_resolution), "--render_depth",
                     "--num_rotations=1", "--only_liver", liver])
    else:
        subprocess.call([blenderPath, "--background", "--python", renderPath, "--", "--blender_files", liverPath,
                         "--rendered_dir", renderedPath, "--rotate_random", "--render_depth",
                         "--num_rotations=2", "--num_samples", str(args.randomRotation), "--only_liver", liver])

    # label images for accuracy checks
    ratingsFilePath = os.path.join(renderedPath, "ratings.txt")
    resultsFilePath = os.path.join(renderedPath, "results.txt")
    rateImages(renderedPath, ratingsFilePath, num_tumors=num_tumors, rotationsfile=resultsFilePath)

    # read matching quaternions
    quaternions = []
    resultsFile = open(resultsFilePath)
    for line in resultsFile:
        arrayLine = line.rstrip().split(" ")
        quaternions.append((arrayLine[1], arrayLine[2], arrayLine[3], arrayLine[4]))

    if args.generate:
        # get best rating and write it to results
        allRatings = [("_".join(line.split()[0].split("/")[0].split("_")[:-1]), line.split()[1]) for line in open(ratingsFilePath)]
        bestRating = sorted(zip(allRatings, quaternions), key = lambda t: t[0][1], reverse=True)[0]
        ratings_view.write(bestRating[0][0] + " " + bestRating[1][0] + " " + bestRating[1][1]  + " "
                                + bestRating[1][2] + " " + bestRating[1][3] + " rating: " + bestRating[0][1] + "\n")
    else:
        # convert data
        converter = RatingRecordConverter(renderedPath, ratingsFilePath, split_in_classes=None, shuffle=False)
        converter.convert()
        recordsFilePath = os.path.join(renderedPath, "ratings_record.tfrecords")

        # predict best image
        predictions, labels = predict_regression.predict(recordsFilePath, num_examples, nntype, 10, no_depth=False, restorePath=args.model_path)
        tf.reset_default_graph()

        sorted_predictions = sorted(zip(predictions, quaternions), key = lambda t: t[0], reverse=True)
        sorted_labels = sorted(zip(labels, quaternions), key = lambda t: t[0], reverse=True)

        allSortedPredictions.append(sorted_predictions)
        allSortedLabels.append(sorted_labels)

        # for p, l in zip(sorted_predictions, sorted_labels)[:5]:
        #     print p, l

        if sorted_labels[0][1] in [p[1] for p in sorted_predictions[:10]]:
            top10Predictions.append(1)
        else:
            top10Predictions.append(0)

        predictedQuaternion = quaternionFromTuple(sorted_predictions[0][1])
        labelQuaternion = quaternionFromTuple(sorted_labels[0][1])

        angle_diff = abs(math.degrees(predictedQuaternion.angle) - math.degrees(labelQuaternion.angle))
        axis_diff = math.degrees(math.acos(np.clip(np.dot(predictedQuaternion.axis, labelQuaternion.axis), -1, 1)))
        angleDiffs.append(angle_diff)
        axisDiffs.append(axis_diff)

# evaluations
avg_angleDiff = np.mean(angleDiffs)
avg_axisDiff = np.mean(axisDiffs)
correct_50 = correctPredictions(angleDiffs, axisDiffs, 50)
correct_top10 = np.mean(top10Predictions)

print "avg angle:", avg_angleDiff
print "avg axis:", avg_axisDiff
print "accuracy (0.50): ", correct_50
print "correct top 10: ", correct_top10

# save results to pickle file
results = {'avg_angleDiff' : avg_angleDiff, 'avg_axisDiff' : avg_axisDiff, 'allSortedPredictions' : allSortedPredictions,
           'allSortedLabels': allSortedLabels}

with open(args.save_file, "wb") as fp:
    pickle.dump(results, fp)