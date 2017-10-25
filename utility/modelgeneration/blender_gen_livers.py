""" generates combinations of liver-models and tumor-models with blender.
    names of liver and tumor-objects have to start with 'liver' and 'tumor'.

    usage: blender --background --python blender_gen_livers.py -- [argument-list]
"""

import bpy
from mathutils import *;
import math
import os
import copy
import argparse
import itertools
import random
import sys


def checkTumorOverlap(tumor1, tumor2):
    """ checks if two tumors are overlapping each other """
    # direction t1 -> t2
    returnValue = False
    t1_cast = getLastRaycastLocation(tumor1, tumor1.location, tumor2.location)
    t2_cast = getFirstRaycastLocation(tumor2, tumor1.location, tumor2.location)
    dist_t1_to_t1 = t1_cast - tumor1.location
    dist_t1_to_t2 = t2_cast - tumor1.location
    if dist_t1_to_t1 > dist_t1_to_t2:
        returnValue = True
    # direction t2 -> t2
    t2_cast = getLastRaycastLocation(tumor2, tumor2.location, tumor1.location)
    t1_cast = getFirstRaycastLocation(tumor1, tumor2.location, tumor1.location)
    dist_t2_to_t2 = t2_cast - tumor2.location
    dist_t2_to_t1 = t1_cast - tumor2.location
    if dist_t2_to_t2 > dist_t2_to_t1:
        returnValue = True
    for obj in bpy.data.objects:
        obj.hide = False
    return returnValue


def noTumorCollision(placedTumors, newTumor):
    """ checks if a newly placed tumor collides with already placed tumors """
    if len(placedTumors) == 0:
        return True
    for tumor in placedTumors:
        overlapping = checkTumorOverlap(bpy.data.objects[tumor], bpy.data.objects[newTumor])
        if overlapping:
            return False
    return True


def getFirstRaycastLocation(object, startLocation, endLocation):
    """ returns the first raycast hit onto an object """
    for obj in bpy.data.objects:
        obj.hide = True
    object.hide = False
    direction = endLocation - startLocation
    scene = bpy.data.scenes[0]
    results = scene.ray_cast(startLocation, direction)
    return results[1]

def hasMinDimensions(tumorName, min_tumor_dim):
    """ checks the minimum size of a tumor. returns True if min_tumor_dim is 0 """
    tumor = bpy.data.objects[tumorName]
    dim_x = tumor.dimensions[0]
    dim_y = tumor.dimensions[1]
    dim_z = tumor.dimensions[2]
    dim_avg = (dim_x + dim_y + dim_z) / 3.0
    return dim_avg >= min_tumor_dim or min_tumor_dim == 0

def scaleTumorUp(tumorName, min_size):
    """ scales tumor up until minimum averageSize is reached """
    while not hasMinDimensions(tumorName, min_size):
        # make tumor 10% bigger in each dimension
        tumor = bpy.data.objects[tumorName]
        tumor.dimensions = tumor.dimensions * 1.10
        bpy.data.scenes[0].update()


# def getLastRaycastLocation(object, startLocation, endLocation, enhanceEnd=False):
#     for obj in bpy.data.objects:
#         obj.hide = True
#     object.hide = False
#     if enhanceEnd:
#         tracingLocation = endLocation * 100
#     else:
#         tracingLocation = endLocation
#     scene = bpy.data.scenes[0]
#     results = scene.ray_cast(startLocation, tracingLocation)
#     newResults = results
#     while newResults[0] != False:
#         # 1.01 offset of old location
#         results = newResults
#         newResults = scene.ray_cast(results[1] * 1.01, tracingLocation)
#     return results[1]

def getLastRaycastLocation(object, startLocation, endLocation):
    """ returns the last last raycast hit onto an object """
    for obj in bpy.data.objects:
        obj.hide = True
    object.hide = False
    direction = endLocation - startLocation
    scene = bpy.data.scenes[0]
    results = scene.ray_cast(startLocation, direction)
    newResults = results
    while newResults[0] != False:
        # 1.01 offset of old location
        results = newResults
        newResults = scene.ray_cast(results[1] + 0.01 * direction, direction)
    return results[1]


def tumorInsideLiver(liver, tumor):
    """ checks if a tumor is placed inside the liver """
    scene = bpy.data.scenes[0]
    # find liver intersection (liver is at (0,0,0))
    liverIntersection = getLastRaycastLocation(liver, liver.location, 10 * tumor.location)
    # find tumor intersection
    tumorIntersection = getLastRaycastLocation(tumor, tumor.location, 10 * tumor.location)
    # calculate distances to liver center
    tumorDistance = (tumorIntersection - liver.location).length
    liverDistance = (liverIntersection - liver.location).length
    for obj in bpy.data.objects:
        obj.hide = False
    return tumorDistance < liverDistance


def randomizeTumor(liver, tumor, placedTumors, min_tumor_dim):
    """ randomly places tumor inside liver while not colliding with other tumors """
    random_seed = 0
    oldLocation = copy.deepcopy(tumor.location)
    oldScale = copy.deepcopy(tumor.scale)
    oldRotation = copy.deepcopy(tumor.rotation_euler)
    # 1000 trys
    for i in range(10000):
        bpy.ops.object.select_all(action='DESELECT')
        tumor.select = True
        if min_tumor_dim:
            scaleTumorUp(tumor.name, min_tumor_dim)
        randomizeTransform(random_seed)
        if tumorInsideLiver(liver, tumor) and noTumorCollision(placedTumors, tumor.name) and hasMinDimensions(
                tumor.name, min_tumor_dim):
            return True
        random_seed += 1
        tumor.location = oldLocation
        tumor.scale = oldScale
        tumor.rotation_euler = oldRotation
    print("couldnt randomly place {0}".format(tumor.name))
    return False


def randomizeTransform(random_seed, scale=(1.0, 1.0, 1.0)):
    bpy.ops.object.randomize_transform(random_seed=random_seed, use_delta=False, use_loc=True,
                                       loc=Vector((100, 100, 100)), use_rot=True, rot=Vector((180, 180, 180)),
                                       use_scale=True, scale_even=True, scale=Vector(scale))


def parseArguments():
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]
    parser = argparse.ArgumentParser(description='Generate Livers')
    parser.add_argument('centered_livers', type=str, help='path to directory with all liver files')
    parser.add_argument('write_to', type=str, help='path to write to')
    parser.add_argument('tumors_file', type=str, help='blender file with all tumors')
    parser.add_argument('num_generated', type=int,
                        help='number of generated liver/tumor combinations per input liver')
    parser.add_argument('--max_tumors', type=int, default=109, help='number of total tumors in tumors_file')
    parser.add_argument('--num_tumors', type=int, default=1, help='numer of tumors per liver')
    parser.add_argument('--random_tumors', type=bool, default=False, help='randomize the chosen tumors')
    parser.add_argument('--min_tumor_dim', type=int, default=0,
                        help='minimum tumor dimension (average of XYZ), not used if 0')
    return parser.parse_args(argv)


def main():
    args = parseArguments()
    centeredLiversDir = args.centered_livers
    writeToDir = args.write_to
    allTumorsFile = args.tumors_file

    dir = os.listdir(centeredLiversDir)
    fileIndex = 0
    for file in dir:
        if "liver" in file:
            fileIndex += 1
            print("start generation for: {0}".format(file))
            bpy.ops.wm.open_mainfile(filepath=os.path.join(centeredLiversDir, file))
            bpy.ops.object.select_all(action='DESELECT')
            # delete all existing tumor objects
            for obj in bpy.data.objects:
                if "tumor" in obj.name:
                    obj.select = True
            bpy.ops.object.delete()
            liver = bpy.data.objects['liver']
            # setup tumors list (assumes tumors are named tumor0, tumor1, ...)
            tumorList = ["tumor{0}".format(i) for i in range(args.max_tumors)]
            if args.random_tumors:
                random.shuffle(tumorList)
            # load every possible combination of tumors
            gen_count = 0
            for tumor_combination in itertools.combinations(tumorList, args.num_tumors):
                if gen_count == args.num_generated:
                    break
                # place tumors randomly
                placedTumors = []
                for tumorName in tumor_combination:
                    bpy.ops.wm.append(filename=os.path.join(allTumorsFile, "Object", tumorName))
                    tumor = bpy.data.objects[tumorName]
                    _tumorInside = randomizeTumor(liver, tumor, placedTumors, args.min_tumor_dim)
                    if _tumorInside:
                        placedTumors.append(tumorName)
                if len(placedTumors) == args.num_tumors:
                    filepath = os.path.join(writeToDir, "gen_liver_{0}_{1}.blend".format(fileIndex, gen_count))
                    bpy.ops.wm.save_as_mainfile(filepath=filepath)
                    gen_count += 1
                    print("liver {0} finished.".format(fileIndex))
                else:
                    print("Could not place tumors in liver {0}".format(fileIndex))
                # remove tumors for next combination
                bpy.ops.object.select_all(action='DESELECT')
                for tumorName in tumor_combination:
                    bpy.data.objects[tumorName].select = True
                bpy.ops.object.delete()


if __name__ == '__main__': main()
