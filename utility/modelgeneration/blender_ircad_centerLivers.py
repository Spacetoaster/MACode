""" sets the center of the coordinate system to the center-of-mass of the liver object (named liver). """

import bpy
from mathutils import *;
import math
import os
  
def centerLiver():
  objects = bpy.data.objects
  liver = objects['liver']
  
  # show all objects and set liver as parent
  for obj in objects:
    obj.hide = False
    if obj != liver:
        obj.parent = liver
        obj.matrix_parent_inverse  = liver.matrix_world.inverted()
  
  # set origins to center of mass
  for obj in objects:
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
  
  # move liver (and rest) to (0,0,0)
  liver.location = Vector((0,0,0))
  
  # clear parent relationship and keep transformation
  for obj in objects:
    obj.select = True
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
  bpy.ops.object.select_all(action='DESELECT')

liverDir = "/Users/Spacetoaster/Desktop/testlivers/blender"
write_To = "/Users/Spacetoaster/Desktop/centered"

dir = os.listdir(liverDir)
for file in dir:
  if "liver" in file:
    print(file)
    bpy.ops.wm.open_mainfile(filepath="/Users/Spacetoaster/Desktop/testlivers/blender/" + file)
    centerLiver()
    bpy.ops.wm.save_as_mainfile(filepath="/Users/Spacetoaster/Desktop/results/" + file)