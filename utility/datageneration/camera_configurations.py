""" camera configurations for rendering """

import bpy

def config_xaxis():
    cameraPositions = []
    cameraPositions.append((500, 0, 0))
    return cameraPositions

def config_2cams():
    cameraPositions = []
    cameraPositions.append((500, 500, 500))
    cameraPositions.append((-500, -500, -500))
    return cameraPositions

def config_3cams():
    cameraPositions = []
    cameraPositions.append((500, 0, 0))
    cameraPositions.append((0, 500, 0))
    cameraPositions.append((0, 0, 500))
    return cameraPositions

def config_6cams():
    cameraPositions = []
    cameraPositions.append((500, 0, 0))
    cameraPositions.append((0, 500, 0))
    cameraPositions.append((0, 0, 500))
    cameraPositions.append((-500, 0, 0))
    cameraPositions.append((0, -500, 0))
    cameraPositions.append((0, 0, -500))
    return cameraPositions

def config_8cams():
    cameraPositions = []
    cameraPositions.append((500, -500, 500))
    cameraPositions.append((500, -500, -500))
    cameraPositions.append((-500, -500, -500))
    cameraPositions.append((-500, -500, 500))
    cameraPositions.append((500, 500, 500))
    cameraPositions.append((500, 500, -500))
    cameraPositions.append((-500, 500, -500))
    cameraPositions.append((-500, 500, 500))
    return cameraPositions

def config_16cams():
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
    return cameraPositions



