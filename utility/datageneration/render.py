import argparse
import math
import os
import random
import sys

import bpy
import numpy as np
from mathutils import *;

pardir = os.path.join(os.path.dirname(__file__), "../../")
dir = os.path.dirname(__file__)
sys.path.append(dir)
sys.path.append(pardir)
print(sys.path)

import render_options
import camera_configurations


class Render:
    """ Renders images with blender, has to be started with blender --background --python /path/to/render.py -- [args] """

    def __init__(self, blenderFilesDir, renderedDir, num_samples, num_rotations, cameraPositions, resolution_x,
                 resolution_y, render_depth=False, copy_rotations=None, icosphere=None, render_options=0,
                 render_bruteforce_ico=None, rotate_random=False, only_liver=None, render_upright=False, only_rotation=None):
        self.blenderFilesDir = blenderFilesDir
        self.renderedDir = renderedDir
        self.num_samples = num_samples
        self.num_rotations = num_rotations
        self.cameraPositions = cameraPositions
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.render_depth = render_depth
        self.rotationsFile = open(os.path.join(renderedDir, "results.txt"), "w+")
        self.copy_rotations = copy_rotations
        self.icosphere = icosphere
        self.render_options = render_options
        self.render_bruteforce_ico = render_bruteforce_ico
        self.rotate_random = rotate_random
        self.only_liver = only_liver
        self.render_upright = render_upright
        self.only_rotation = only_rotation

    def renderImages(self, fileName, renderLocation):
        # order is important!
        # number of images depends only on number of camera Positions!
        for i, camPos in enumerate(self.cameraPositions):
            self.hideInRenderingExcept("")
            self.enableColorRendering()
            scene.render.filepath = "{0}/{1}_{2}.png".format(renderLocation, fileName, i)
            camera.location = Vector(camPos)
            print(camera.location)
            print(bpy.context.scene.render.engine)
            # render color image
            bpy.ops.render.render(write_still=True)
            if self.render_depth:
                # render depth image
                self.renderDepthImages(fileName, renderLocation, i)

    def initialize(self):
        # add camera
        bpy.ops.object.camera_add()
        global camera
        camera = bpy.data.objects['Camera']
        global liver
        liver = bpy.data.objects['liver']
        global scene
        scene = bpy.data.scenes[0]
        bpy.ops.object.empty_add(type='ARROWS', location=(0, 0, 0))
        # set liver rotation to (0,0,0)
        self.makeLiverParent()
        self.setLiverRotation(Quaternion((1.0, 0.0, 0.0, 0.0)))
        # constraint: track to empty (0,0,0) with rotation of liver (up vector is z-axis)
        empty = bpy.data.objects['Empty']
        empty.rotation_euler = liver.rotation_euler.copy()
        constraint = camera.constraints.new(type='TRACK_TO')
        constraint.target = empty
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
        # constraint.use_target_z = True
        # camera and rendering settings
        bpy.context.scene.camera = camera
        bpy.data.cameras['Camera'].type = 'ORTHO'
        # bpy.data.cameras['Camera'].ortho_scale = 500
        bpy.data.cameras['Camera'].clip_start = 0
        bpy.data.cameras['Camera'].clip_end = 5000
        # maxCoodinate = max([abs(a) for x in liver.bound_box for a in x])
        maxCoordinate = 0
        for node in liver.bound_box:
            length = Vector((node[0], node[1], node[2])).length
            if length > maxCoordinate:
                maxCoordinate = length

        bpy.data.cameras['Camera'].ortho_scale = 2 * maxCoordinate
        scene.render.resolution_percentage = 100
        scene.render.resolution_x = self.resolution_x
        scene.render.resolution_y = self.resolution_y

    def hideInRenderingExcept(self, nameexception):
        for object in bpy.data.objects:
            object.hide_render = nameexception not in object.name and object.type == 'MESH'

    def enableDepthRendering(self):
        scene = bpy.data.scenes[0]
        scene.use_nodes = True
        tree = scene.node_tree
        links = tree.links
        renderLayers = tree.nodes['Render Layers']
        composite = tree.nodes['Composite']
        links.new(renderLayers.outputs['Z'], composite.inputs['Image'])
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
        scene.render.use_antialiasing = False

    def enableColorRendering(self):
        scene.use_nodes = False
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_depth = "8"
        scene.render.use_antialiasing = True

    def renderDepthImages(self, fileName, renderLocation, i):
        self.enableDepthRendering()
        # render Tumors
        self.hideInRenderingExcept("tumor")
        scene.render.filepath = "{0}/{1}_{2}_dt.jpg".format(renderLocation, fileName, i)
        bpy.ops.render.render(write_still=True)
        # render Liver
        self.hideInRenderingExcept("liver")
        scene.render.filepath = "{0}/{1}_{2}_dl.jpg".format(renderLocation, fileName, i)
        bpy.ops.render.render(write_still=True)

    def setRenderOptimizations(self):
        bpy.data.scenes['Scene'].render.use_raytrace = False
        # bpy.data.scenes['Scene'].render.use_shadows = False
        # bpy.data.scenes['Scene'].render.use_antialiasing = False
        bpy.data.scenes['Scene'].render.tile_x = 100
        bpy.data.scenes['Scene'].render.tile_y = 100
        # bpy.data.scenes['Scene'].render.preview_start_resolution = 100

    def setLiverRotation(self, newRotation):
        objects = bpy.data.objects
        liver = objects['liver']
        bpy.ops.object.select_all(action='DESELECT')
        liver.select = True
        liver.rotation_mode = 'QUATERNION'
        liver.rotation_quaternion = newRotation

    def makeLiverParent(self):
        objects = bpy.data.objects
        liver = objects['liver']
        for obj in objects:
            obj.hide = False
            if obj != liver and obj != bpy.data.objects['Camera'] and obj != bpy.data.objects['Empty']:
                obj.parent = liver
                obj.matrix_parent_inverse = liver.matrix_world.inverted()

    def getRandomSphereVector(self):
        u = random.uniform(0, 1)
        v = random.uniform(0, 1)
        phi = 2 * math.pi * u
        theta = math.acos(2 * v - 1)
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        return x, y, z

    def getNormalizedQuaternion(self, angle, x, y, z):
        qx = x * math.sin(math.radians(angle / 2))
        qy = y * math.sin(math.radians(angle / 2))
        qz = z * math.sin(math.radians(angle / 2))
        w = math.cos(math.radians(angle / 2))
        quat = Quaternion((w, qx, qy, qz))
        quat.normalize()
        return quat

    def getCopiedRotationsDict(self):
        rotationsFile = open(self.copy_rotations)

        rotationsDict = {}
        for line in rotationsFile:
            arrayLine = line.rstrip().split(" ")
            blenderFile = arrayLine[0][:arrayLine[0].rfind("_")]
            if not blenderFile in rotationsDict:
                rotationsDict[blenderFile] = []
            quaternion = Quaternion(
                (float(arrayLine[1]), float(arrayLine[2]), float(arrayLine[3]), float(arrayLine[4])))
            rotationsDict[blenderFile].append(quaternion)
        return rotationsDict

    def setRenderOptions(self):
        if self.render_options == 1:
            render_options.setMaterials_translucent_shaded()
        elif self.render_options == 2:
            render_options.setMaterials_opaque_shaded()
        else:
            render_options.setMaterials_translucent_shadeless()

    def render(self):
        centeredLiversDir = self.blenderFilesDir

        if not os.path.exists(self.renderedDir):
            os.mkdir(self.renderedDir)

        dir = os.listdir(centeredLiversDir)
        blenderFiles = ["liver1.blend"]
        allLivers = ["liver"]
        blenderFiles = allLivers
        rotateRandom = self.rotate_random

        if self.only_liver:
            blenderFiles = [self.only_liver]

        rotationsDict = None
        if self.copy_rotations:
            rotationsDict = self.getCopiedRotationsDict()

        for file in dir:
            if any(x in file for x in blenderFiles):
                bpy.ops.wm.open_mainfile(filepath=os.path.join(centeredLiversDir, file))
                self.initialize()
                self.setRenderOptimizations()
                self.setRenderOptions()
                rotations = np.linspace(0, 180, self.num_rotations, endpoint=False)
                if self.icosphere:
                    self.renderIcosphere(file, rotations)
                elif self.render_bruteforce_ico:
                    self.renderBruteforceIco(file)
                elif self.render_upright:
                    self.renderUpright(file)
                elif self.only_rotation:
                    self.renderOnlyRotation(file)
                else:
                    self.renderRandom(file, rotateRandom, rotationsDict, rotations)
        self.rotationsFile.close()

    def renderOnlyRotation(self, file):
        rotation = Quaternion((self.only_rotation[0], self.only_rotation[1], self.only_rotation[2], self.only_rotation[3]))
        self.setLiverRotation(rotation)
        self.renderAndWrite(file, self.only_rotation[4], rotation, (0.0, 0.0, 0.0), 0.0)

    def renderUpright(self, file):
        rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
        rotation.normalize()
        self.setLiverRotation(rotation)
        self.renderAndWrite(file, None, rotation, (0.0, 0.0, 0.0), 0.0)

    def renderRandom(self, file, rotateRandom, rotationsDict, rotations):
        renderNumber = 0
        for i in range(self.num_samples):
            x, y, z = self.getRandomSphereVector()
            for rotation in rotations[1:]:
                if (rotateRandom):
                    rotation = random.uniform(0, 180)
                newRotation = self.getNormalizedQuaternion(rotation, x, y, z)
                if self.copy_rotations:
                    newRotation = rotationsDict[file].pop(0)
                self.setLiverRotation(newRotation)
                self.renderAndWrite(file, renderNumber, newRotation, (x, y, z), rotation)
                renderNumber += 1

    def renderIcosphere(self, file, rotations):
        # get vertices of Icosphere
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=self.icosphere, location=(0.0, 0.0, 0.0))
        vertexList = []
        for vertex in bpy.data.objects['Icosphere'].data.vertices:
            vertexList.append(vertex.co.copy())
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['Icosphere'].select = True
        bpy.ops.object.delete()
        # walk through vertexList
        renderNumber = 0
        for vertex in vertexList:
            vertex.normalize()
            x, y, z = vertex.to_tuple()
            for rotation in rotations[1:]:
                newRotation = self.getNormalizedQuaternion(rotation, x, y, z)
                self.setLiverRotation(newRotation)
                self.renderAndWrite(file, renderNumber, newRotation, (x, y, z), rotation)
                renderNumber += 1

    def quat_rotation_between_vectors(self, v1, v2):
        """ returns the quaternion required to rotate v1 to v2 """
        if not np.array_equal(np.multiply(v1, -1), v2):
            qw = np.dot(v1, v2) + math.sqrt(np.square(np.linalg.norm(v1)) * np.square(np.linalg.norm(v2)))
            qx = np.cross(v1, v2)[0]
            qy = np.cross(v1, v2)[1]
            qz = np.cross(v1, v2)[2]
            quat = Quaternion((qw, qx, qy, qz)).normalized()
        else:
            quat = self.getNormalizedQuaternion(180, 0.0, 0.0, 1.0)
        return quat

    def renderBruteforceIco(self, file):
        """ rotate x-axis of object to all vertices of icosphere """
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=self.render_bruteforce_ico, location=(0.0, 0.0, 0.0))
        vertexList = []
        for vertex in bpy.data.objects['Icosphere'].data.vertices:
            vertexList.append(vertex.co.copy())
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['Icosphere'].select = True
        bpy.ops.object.delete()
        renderNumber = 0
        for vertex in vertexList:
            vertex.normalize()
            x, y, z = vertex.to_tuple()
            self.setLiverRotation(Quaternion((1.0, 0.0, 0.0, 0.0)))
            rotation = self.quat_rotation_between_vectors([1.0, 0.0, 0.0], [x, y, z])
            self.setLiverRotation(rotation)
            self.renderAndWrite(file, renderNumber, rotation, (x, y, z), 0)
            renderNumber += 1

    def renderAndWrite(self, file, renderNumber, newRotation, vector, angle):
        if renderNumber != None:
            renderLocation = "{0}/{1}_{2}".format(self.renderedDir, file, renderNumber)
        else:
            renderLocation = "{0}/{1}".format(self.renderedDir, file)
        if not os.path.exists(renderLocation):
            os.mkdir(renderLocation)
        self.renderImages('liver', renderLocation)
        if renderNumber != None:
            name = "{0}_{1}".format(file, renderNumber)
        else:
            name = file
        line = name + " " + str(newRotation[0]) + " " + str(
            newRotation[1]) + " " + str(newRotation[2]) + " " + str(newRotation[3])
        line += " vector({0}, {1}, {2})".format(vector[0], vector[1], vector[2]) + " " + str(angle)
        line += "\n"
        self.rotationsFile.write(line)


def parseArguments():
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]
    parser = argparse.ArgumentParser(description='Render Input-Images')
    parser.add_argument('--blender_files', type=str, help='folder with blender files to render from')
    parser.add_argument('--only_liver', type=str, default=None, help='render only liver with specified name')
    parser.add_argument('--only_rotation', nargs='+', type=float, default=None)
    parser.add_argument('--rendered_dir', type=str, help='folder to render to')
    parser.add_argument('--num_samples', type=int, help='number of samples')
    parser.add_argument('--num_rotations', type=int, help='number of rotations')
    parser.add_argument('--icosphere', type=int, default=None,
                        help='if specified uses icosphere rotations (deterministic) with the specified amount of subdivision')
    parser.add_argument('--res_x', type=int, default=100, help='resolution x')
    parser.add_argument('--res_y', type=int, default=100, help='resolution y')
    parser.add_argument('--render_depth', action='store_true', default=False,
                        help='set to render depth images of tumors and liver')
    parser.add_argument('--copy_rotations', type=str, default=None,
                        help='file to copy rotations from, in order to render the same data with different parameters')
    parser.add_argument('--render_options', type=int, default=0,
                        help='0 translucent shadeless, 1 translucent shaded, 2 opaque shaded')
    parser.add_argument('--num_cams', type=int, default=None)
    parser.add_argument('--render_bruteforce_ico', type=int, default=None)
    parser.add_argument('--rotate_random', dest='rotate_random', action='store_true', default=False)
    parser.add_argument('--render_upright', dest='render_upright', action='store_true', default=False)
    # parser.add_argument('--rate_images', type=bool, default=False, help='set to True to add image ratings in results')
    # parser.add_argument('--num_tumors', type=int, default=0, help='number of tumors for image rating')
    return parser.parse_args(argv)


def main():
    args = parseArguments()

    if args.num_cams == 2:
        cameraPositions = camera_configurations.config_2cams()
    elif args.num_cams == 3:
        cameraPositions = camera_configurations.config_3cams()
    elif args.num_cams == 6:
        cameraPositions = camera_configurations.config_6cams()
    elif args.num_cams == 8:
        cameraPositions = camera_configurations.config_8cams()
    elif args.num_cams == 16:
        cameraPositions = camera_configurations.config_16cams()
    else:
        cameraPositions = camera_configurations.config_xaxis()

    render = Render(args.blender_files, args.rendered_dir, args.num_samples, args.num_rotations, cameraPositions,
                    args.res_x, args.res_y, args.render_depth, args.copy_rotations, args.icosphere, args.render_options,
                    args.render_bruteforce_ico, args.rotate_random, args.only_liver, args.render_upright, args.only_rotation)
    render.render()


if __name__ == '__main__': main()
