""" evaluates results of sampling method (bruteforce) """

import numpy as np
import argparse
import pickle
from pyquaternion import Quaternion
import math

parser = argparse.ArgumentParser(description='Bruteforce')
parser.add_argument('--load_path', type=str, default=None)
args = parser.parse_args()

saveFile = open(args.load_path, "rb")
results = pickle.load(saveFile)

otherSaveFile = open("bf_run_ico162", "rb")
results162 = pickle.load(otherSaveFile)

deviations = []
predicted_worse = 0
predicted_better = 0
prediction_deviation = []

angle_diffs = []
axis_diffs = []

def quaternionFromTuple(tuple):
    qw = float(tuple[0])
    qx = float(tuple[1])
    qy = float(tuple[2])
    qz = float(tuple[3])
    return Quaternion(qw, qx, qy, qz)

def quatDiff(tuple1, tuple2):
    q1 = quaternionFromTuple(tuple1)
    q2 = quaternionFromTuple(tuple2)
    angle_diff = abs(math.degrees(q1.angle) - math.degrees(q2.angle))
    axis_diff = math.degrees(math.acos(np.clip(np.dot(q1.axis, q2.axis), -1, 1)))
    return angle_diff, axis_diff


for i in range(len(results['allSortedPredictions'])):
    sortedPredictions = results['allSortedPredictions'][i]
    sortedLabels = results['allSortedLabels'][i]
    index = [x[1] for x in sortedLabels].index(sortedPredictions[0][1])
    bestViewRating = sortedLabels[0][0]
    labelOfBestPrediction = sortedLabels[index][0]
    prediction = sortedPredictions[0][0]

    actualBestRating = results162['allSortedLabels'][i][0][0]

    angleDiff, axisDiff = quatDiff(results162['allSortedLabels'][i][0][1],
                                   sortedPredictions[0][1])
    angle_diffs.append(angleDiff)
    axis_diffs.append(axisDiff)

    deviation = abs(labelOfBestPrediction - actualBestRating)
    deviations.append(deviation)
    if labelOfBestPrediction >= actualBestRating:
        predicted_better += 1
    else:
        predicted_worse += 1
    prediction_deviation.append(abs(bestViewRating - prediction))
    # quaternion stuff

# print "avg axis diff", results['avg_axisDiff']
# print "avg angle diff", results['avg_angleDiff']
print "avg axis diff", np.mean(axis_diffs)
print "avg angle diff", np.mean(angle_diffs)
print "avg deviation: ", np.mean(deviations)
print "predicted worse: ", predicted_worse
print "predicted better: ", predicted_better
print "predicted_deviation: ", np.mean(prediction_deviation)
