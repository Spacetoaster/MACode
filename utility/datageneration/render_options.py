""" render options for rendering """

import bpy

def setMaterials_translucent_shadeless():
    """ all objects rendered shadeless (without shadows/specularity), all translucent """
    bpy.ops.object.lamp_add(type='HEMI')
    liverMaterial = bpy.data.materials.new('LiverMaterial')
    liverMaterial.diffuse_color = (1.0, 0.0, 0.0)
    liverMaterial.use_transparency = True
    liverMaterial.use_shadeless = True  # no shadows, no specular light
    liverMaterial.alpha = 0.5
    liverObjects = [x for x in bpy.data.objects if "liver" in x.name]
    for o in liverObjects:
        o.active_material = liverMaterial

    passiveMaterial = bpy.data.materials.new('PassiveMaterial')
    passiveMaterial.diffuse_color = (0.0, 1.0, 0.0)
    passiveMaterial.use_transparency = True
    passiveMaterial.use_shadeless = True  # no shadows, no specular light
    passiveMaterial.alpha = 0.5
    passive = [x for x in bpy.data.objects if
               "liver" not in x.name and "Camera" not in x.name and "tumor" not in x.name and "Empty" not in x.name and "Hemi" not in x.name]
    for o in passive:
        o.active_material = passiveMaterial

    tumorMaterial = bpy.data.materials.new('tumorMaterial')
    tumorMaterial.diffuse_color = (0.0, 0.0, 1.0)
    tumorMaterial.use_shadeless = True  # no shadows, no specular light
    tumorMaterial.use_transparency = True
    tumorMaterial.alpha = 0.5
    tumors = [x for x in bpy.data.objects if "tumor" in x.name]
    for o in tumors:
        o.active_material = tumorMaterial

def setMaterials_translucent_shaded():
    """ all objects rendered shaded, all translucent """
    bpy.ops.object.lamp_add(type='HEMI')
    bpy.data.lamps['Hemi'].energy = 0.5
    bpy.ops.object.lamp_add(type='POINT')
    bpy.data.objects['Point'].location = (0, 500, 0)
    bpy.data.lamps['Point'].falloff_type = 'CONSTANT'
    liverMaterial = bpy.data.materials.new('LiverMaterial')
    liverMaterial.diffuse_color = (1.0, 0.0, 0.0)
    liverMaterial.use_transparency = True
    liverMaterial.alpha = 0.5
    liverObjects = [x for x in bpy.data.objects if "liver" in x.name]
    for o in liverObjects:
        o.active_material = liverMaterial

    passiveMaterial = bpy.data.materials.new('PassiveMaterial')
    passiveMaterial.diffuse_color = (0.0, 1.0, 0.0)
    passiveMaterial.use_transparency = True
    passiveMaterial.alpha = 0.5
    passive = [x for x in bpy.data.objects if
               "liver" not in x.name and "Camera" not in x.name and "tumor" not in x.name and "Empty" not in x.name and "Hemi" not in x.name]
    for o in passive:
        o.active_material = passiveMaterial

    tumorMaterial = bpy.data.materials.new('tumorMaterial')
    tumorMaterial.diffuse_color = (0.0, 0.0, 1.0)
    tumorMaterial.use_transparency = True
    tumorMaterial.alpha = 0.5
    tumors = [x for x in bpy.data.objects if "tumor" in x.name]
    for o in tumors:
        o.active_material = tumorMaterial

def setMaterials_opaque_shaded():
    """ all objects rendered shaded, all translucent """
    bpy.ops.object.lamp_add(type='HEMI')
    bpy.data.lamps['Hemi'].energy = 0.5
    bpy.ops.object.lamp_add(type='POINT')
    bpy.data.objects['Point'].location = (0, 500, 0)
    bpy.data.lamps['Point'].falloff_type = 'CONSTANT'
    liverMaterial = bpy.data.materials.new('LiverMaterial')
    liverMaterial.diffuse_color = (1.0, 0.0, 0.0)
    liverObjects = [x for x in bpy.data.objects if "liver" in x.name]
    for o in liverObjects:
        o.active_material = liverMaterial

    passiveMaterial = bpy.data.materials.new('PassiveMaterial')
    passiveMaterial.diffuse_color = (0.0, 1.0, 0.0)
    passive = [x for x in bpy.data.objects if
               "liver" not in x.name and "Camera" not in x.name and "tumor" not in x.name and "Empty" not in x.name and "Hemi" not in x.name]
    for o in passive:
        o.active_material = passiveMaterial

    tumorMaterial = bpy.data.materials.new('tumorMaterial')
    tumorMaterial.diffuse_color = (0.0, 0.0, 1.0)
    tumors = [x for x in bpy.data.objects if "tumor" in x.name]
    for o in tumors:
        o.active_material = tumorMaterial


def setMaterials_opaque_passive():
    """ passive material is rendered opague, rest is shadeless """
    bpy.ops.object.lamp_add(type='HEMI')
    liverMaterial = bpy.data.materials.new('LiverMaterial')
    liverMaterial.diffuse_color = (1.0, 0.0, 0.0)
    liverMaterial.use_transparency = True
    liverMaterial.use_shadeless = True  # no shadows, no specular light
    liverMaterial.alpha = 0.5
    liverObjects = [x for x in bpy.data.objects if "liver" in x.name]
    for o in liverObjects:
        o.active_material = liverMaterial
    passiveMaterial = bpy.data.materials.new('PassiveMaterial')
    passiveMaterial.diffuse_color = (0.0, 1.0, 0.0)
    passiveMaterial.use_shadeless = True  # no shadows, no specular light
    passive = [x for x in bpy.data.objects if
               "liver" not in x.name and "Camera" not in x.name and "tumor" not in x.name and "Empty" not in x.name and "Hemi" not in x.name]
    for o in passive:
        o.active_material = passiveMaterial
    tumorMaterial = bpy.data.materials.new('tumorMaterial')
    tumorMaterial.diffuse_color = (0.0, 0.0, 1.0)
    tumorMaterial.use_shadeless = True  # no shadows, no specular light
    tumorMaterial.use_transparency = True
    tumorMaterial.alpha = 0.5
    tumors = [x for x in bpy.data.objects if "tumor" in x.name]
    for o in tumors:
        o.active_material = tumorMaterial