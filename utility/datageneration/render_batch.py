""" runs multiple render operations (can also be done with a bash-script)."""

import bpy
import sys
import os
import argparse

# dir = os.path.dirname(sys.argv[3])
dir = os.path.dirname(__file__)
if not dir in sys.path:
    sys.path.append(dir)

from render import Render


def parseArguments():
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]
    parser = argparse.ArgumentParser(description='Render Batch')
    parser.add_argument('--blender_files', type=str, help='folder with blender files to render from')
    parser.add_argument('--rendered_dir', type=str, help='folder to render batch to')
    parser.add_argument('--num_samples_train', type=int, help='number of test samples')
    parser.add_argument('--num_samples_test', type=int, help='number of training samples')
    parser.add_argument('--num_rotations_train', type=int, help='number of rotations in train set')
    parser.add_argument('--num_rotations_test', type=int, help='number of rotations in test set')
    parser.add_argument('--res_x', type=int, default=100, help='resolution x')
    parser.add_argument('--res_y', type=int, default=100, help='resolution y')
    return parser.parse_args(argv)


def renderTrainAndTestSet(blenderFiles, renderedDir, num_samples_train, num_samples_test, num_rotations_train,
                          num_rotations_test, res_x, res_y, cameraPositions):
    global renderList
    renderedDir = os.path.abspath(renderedDir)
    renderList.append(
        Render(blenderFiles, renderedDir + "_train", num_samples_train, num_rotations_train,
               cameraPositions, res_x, res_y))
    renderList.append(
        Render(blenderFiles, renderedDir + "_test", num_samples_test, num_rotations_test, cameraPositions,
               res_x, res_y))


renderList = []


def main():
    args = parseArguments()

    # 2 cams
    cameraPositions = []
    cameraPositions.append((500, 500, 500))
    cameraPositions.append((-500, -500, -500))
    renderTrainAndTestSet(args.blender_files, os.path.join(args.rendered_dir, "liverAll2Cams"), args.num_samples_train,
                          args.num_samples_test, args.num_rotations_train, args.num_rotations_test, args.res_x,
                          args.res_y, cameraPositions)

    # XYZ (3 Cameras)
    cameraPositions = []
    cameraPositions.append((500, 0, 0))
    cameraPositions.append((0, 500, 0))
    cameraPositions.append((0, 0, 500))
    renderTrainAndTestSet(args.blender_files, os.path.join(args.rendered_dir, "liverAll3Cams"), args.num_samples_train,
                          args.num_samples_test, args.num_rotations_train, args.num_rotations_test, args.res_x,
                          args.res_y, cameraPositions)

    # XYZ-X-Y-Z (6 Cameras)
    cameraPositions = []
    cameraPositions.append((500, 0, 0))
    cameraPositions.append((0, 500, 0))
    cameraPositions.append((0, 0, 500))
    cameraPositions.append((-500, 0, 0))
    cameraPositions.append((0, -500, 0))
    cameraPositions.append((0, 0, -500))
    renderTrainAndTestSet(args.blender_files, os.path.join(args.rendered_dir, "liverAll6Cams"), args.num_samples_train,
                          args.num_samples_test, args.num_rotations_train, args.num_rotations_test, args.res_x,
                          args.res_y, cameraPositions)

    # Corners (8 Cameras)
    cameraPositions = []
    cameraPositions.append((500, -500, 500))
    cameraPositions.append((500, -500, -500))
    cameraPositions.append((-500, -500, -500))
    cameraPositions.append((-500, -500, 500))
    cameraPositions.append((500, 500, 500))
    cameraPositions.append((500, 500, -500))
    cameraPositions.append((-500, 500, -500))
    cameraPositions.append((-500, 500, 500))
    renderTrainAndTestSet(args.blender_files, os.path.join(args.rendered_dir, "liverAll8Cams"), args.num_samples_train,
                          args.num_samples_test, args.num_rotations_train, args.num_rotations_test, args.res_x,
                          args.res_y, cameraPositions)

    # Corners and up/down (10 Cameras)
    cameraPositions = []
    cameraPositions.append((500, -500, 500))
    cameraPositions.append((500, -500, -500))
    cameraPositions.append((-500, -500, -500))
    cameraPositions.append((-500, -500, 500))
    cameraPositions.append((500, 500, 500))
    cameraPositions.append((500, 500, -500))
    cameraPositions.append((-500, 500, -500))
    cameraPositions.append((-500, 500, 500))
    cameraPositions.append((0, -500, 0))
    cameraPositions.append((0, 500, 0))
    renderTrainAndTestSet(args.blender_files, os.path.join(args.rendered_dir, "liverAll10Cams"), args.num_samples_train,
                          args.num_samples_test, args.num_rotations_train, args.num_rotations_test, args.res_x,
                          args.res_y, cameraPositions)

    # Corners and inbetween (16 Cameras)
    cameraPositions = []
    cameraPositions.append((500, -500, 500))
    cameraPositions.append((500, -500, 0))
    cameraPositions.append((500, -500, -500))
    cameraPositions.append((0, -500, -500))
    cameraPositions.append((-500, -500, -500))
    cameraPositions.append((-500, -500, 0))
    cameraPositions.append((-500, -500, 500))
    cameraPositions.append((0, -500, 500))
    cameraPositions.append((500, 500, 500))
    cameraPositions.append((500, 500, 0))
    cameraPositions.append((500, 500, -500))
    cameraPositions.append((0, 500, -500))
    cameraPositions.append((-500, 500, -500))
    cameraPositions.append((-500, 500, 0))
    cameraPositions.append((-500, 500, 500))
    cameraPositions.append((0, 500, 500))
    renderTrainAndTestSet(args.blender_files, os.path.join(args.rendered_dir, "liverAll16Cams"), args.num_samples_train,
                          args.num_samples_test, args.num_rotations_train, args.num_rotations_test, args.res_x,
                          args.res_y, cameraPositions)

    # Corners and inbetween (18 Cameras)
    cameraPositions = []
    cameraPositions.append((500, -500, 500))
    cameraPositions.append((500, -500, 0))
    cameraPositions.append((500, -500, -500))
    cameraPositions.append((0, -500, -500))
    cameraPositions.append((-500, -500, -500))
    cameraPositions.append((-500, -500, 0))
    cameraPositions.append((-500, -500, 500))
    cameraPositions.append((0, -500, 500))
    cameraPositions.append((500, 500, 500))
    cameraPositions.append((500, 500, 0))
    cameraPositions.append((500, 500, -500))
    cameraPositions.append((0, 500, -500))
    cameraPositions.append((-500, 500, -500))
    cameraPositions.append((-500, 500, 0))
    cameraPositions.append((-500, 500, 500))
    cameraPositions.append((0, 500, 500))
    cameraPositions.append((0, -500, 0))
    cameraPositions.append((0, 500, 0))
    renderTrainAndTestSet(args.blender_files, os.path.join(args.rendered_dir, "liverAll18Cams"), args.num_samples_train,
                          args.num_samples_test, args.num_rotations_train, args.num_rotations_test, args.res_x,
                          args.res_y, cameraPositions)

    for render in renderList:
        render.render()


if __name__ == '__main__': main()
