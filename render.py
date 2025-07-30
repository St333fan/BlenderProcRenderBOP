import blenderproc as bproc
import bmesh
import bpy
import numpy as np
import os
import glob
import mathutils
from mathutils import Vector
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='BlenderProc object rendering pipeline')
    parser.add_argument('--object_folder', type=str, default='./examples/basics/camera_object_pose',
                        help='Path to folder containing 3D objects')
    parser.add_argument('--num_cameras', type=int, default=10,
                        help='Number of cameras for horizontal, top, and bottom views')
    parser.add_argument('--surface_type', type=str, choices=['textured', 'generated'], default='textured',
                        help='Type of surface to use: textured (from files) or generated (procedural)')
    parser.add_argument('--texture_folder', type=str, default='/home/st3fan/Downloads/Plastic010_4K-JPG/',
                        help='Path to folder containing texture files (only used if surface_type is textured)')
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory for rendered images and annotations')
    
    args = parser.parse_args()

    # Initialize BlenderProc
    bproc.init()

    bproc.renderer.set_render_devices(use_only_cpu=False, desired_gpu_device_type=["CUDA"], desired_gpu_ids=[0])
    
    #bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    
    # Use a very low sample count for speed and high for quality
    bpy.context.scene.cycles.samples = 16

    # Use the GPU-accelerated denoiser to clean up the noisy image
    bproc.renderer.set_denoiser("OPTIX")

    # Configuration - using command line arguments
    OBJECT_FOLDER = args.object_folder
    OUTPUT_DIR = args.output
    RESOLUTION_X = 2000
    RESOLUTION_Y = 1500  # 4:3 aspect ratio in Full HD
    NUM_CAMERAS_HORIZONTAL = args.num_cameras
    NUM_CAMERAS_TOP = args.num_cameras
    NUM_CAMERAS_BOTTOM = args.num_cameras
    CAMERA_ANGLE = 45  # degrees
    OBJECT_COVERAGE = 1/2  # Object should fill 2/3 of camera frame height

    # Set camera resolution
    bproc.camera.set_resolution(RESOLUTION_X, RESOLUTION_Y)
    
    # Enable depth rendering
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    
    # Setup lighting
    setup_lighting()
    
    # Create surface based on specified type
    if args.surface_type == 'textured':
        surface = create_textured_surface(args.texture_folder)
    else:
        surface = create_structured_surface()
    
    # Get all object files from folder
    object_files = []
    for ext in ['*.obj', '*.ply', '*.fbx', '*.dae', '*.blend']:
        object_files.extend(glob.glob(os.path.join(OBJECT_FOLDER, ext)))
    
    if not object_files:
        print(f"No objects found in {OBJECT_FOLDER}")
        return
    
    # Process each object
    for obj_idx, obj_file in enumerate(object_files):
        print(f"Processing object {obj_idx + 1}/{len(object_files)}: {os.path.basename(obj_file)}")
        
        
        # Load object, object scaled from mm dm for better scaling, e.g. input BOP100mm-Blender100mm but here BOP100mm-Blender100cm
        try:
            if obj_file.endswith('.obj'):
                obj = bproc.loader.load_obj(obj_file)[0]
            elif obj_file.endswith('.ply'):
                obj = bproc.loader.load_obj(obj_file)[0]
                obj.set_scale([0.01, 0.01, 0.01])  # Convert mm to meters depth will be saved as 5000 = 50cm
            elif obj_file.endswith('.blend'):
                obj = bproc.loader.load_blend(obj_file)
            else:
                continue
        except Exception as e:
            print(f"Failed to load {obj_file}: {e}")
            continue
        
        obj.set_cp("category_id", obj_idx)  # category_id starts from 0
        
        # Extract the file name without extension
        base_name = os.path.splitext(os.path.basename(obj_file))[0]
    
        # Use this as the folder name
        obj_base_folder = os.path.join(OUTPUT_DIR, base_name)
        
        # Position object at origin and on surface
        position_object_on_surface(obj, surface)
        
        # Get object dimensions for camera distance calculation
        obj_dimensions = get_object_dimensions(obj)
        max_dimension = max(obj_dimensions)
        
        # Calculate camera distance based on object size
        camera_distance = calculate_camera_distance(max_dimension, OBJECT_COVERAGE)

        append = obj_idx != 0  # False for first object, True afterwards
        
        # Phase 1: Horizontal cameras around object + surface
        print("Rendering horizontal cameras with surface...")
        render_horizontal_cameras(obj, camera_distance, NUM_CAMERAS_HORIZONTAL, 
                                obj_idx, "horizontal_with_surface", obj_base_folder, append)

        # Phase 2: Top cameras (45 degrees down)
        print("Rendering top cameras with surface...")
        render_top_cameras(obj, camera_distance, NUM_CAMERAS_TOP, CAMERA_ANGLE,
                          obj_idx, "top_with_surface", obj_base_folder)
        
        # Phase 3: Remove surface and render bottom cameras
        print("Removing surface and rendering bottom cameras...")
        make_surface_transparent(surface)

        render_bottom_cameras(obj, camera_distance, NUM_CAMERAS_BOTTOM, CAMERA_ANGLE,
                    obj_idx, "bottom_no_surface", obj_base_folder)
                   
                  
        # Clear previous objects (except surface)
        clear_scene_objects()
        
        # Recreate surface and lighting for next object
        if obj_idx < len(object_files) - 1:
            if args.surface_type == 'textured':
                surface = create_textured_surface(args.texture_folder)
            else:
                surface = create_structured_surface()
            setup_lighting()  # Refresh lighting setup

def setup_lighting():
    """
    Places a soft area light on each of the 6 sides of a bounding box
    around the origin, creating a "light box" effect for even illumination.
    """
    # Clear existing lights
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()

    # Define the properties for the lights
    distance = 10.0  # How far the lights are from the origin
    energy = 1500   # Brightness of each light
    size = 20      # The size of the area light, determines shadow softness
    color = (1.0, 1.0, 1.0) # Pure white light

    # A list of tuples: (location, rotation) for each of the 6 lights
    light_positions = [
        # Top light, pointing down
        ([0, 0, distance], [np.radians(180), 0, 0]),
        # Bottom light, pointing up
        ([0, 0, -distance], [0, 0, 0]),
        # Front light, pointing towards the back
        ([0, distance, 0], [np.radians(-90), 0, 0]),
        # Back light, pointing towards the front
        ([0, -distance, 0], [np.radians(90), 0, 0]),
        # Right light, pointing left
        ([distance, 0, 0], [0, np.radians(90), 0]),
        # Left light, pointing right
        ([-distance, 0, 0], [0, np.radians(-90), 0])
    ]

    # Create a light for each position and rotation
    for location, rotation in light_positions:
        light = bproc.types.Light()
        light.set_type("AREA")
        light.set_location(location)
        light.set_rotation_euler(rotation)
        light.set_energy(energy)
        light.set_color(color)
        
        # Access the underlying blender object to set the area light's size
        light.blender_obj.data.size = size

    print(f"Lighting setup complete: 6 soft area lights placed.")

def create_structured_surface():
    """Create a structured surface (Oberfläche) with some texture/pattern"""
    # Create a plane
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    surface = bpy.context.active_object
    surface.name = "surface"
    
    # Add subdivision for structure
    bpy.context.view_layer.objects.active = surface
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Subdivide the plane multiple times for detail
    bpy.ops.mesh.subdivide(number_cuts=10)
    bpy.ops.mesh.subdivide(number_cuts=5)
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Manual displacement using bmesh for compatibility
    import bmesh
    
    # Create bmesh representation
    bm = bmesh.new()
    bm.from_mesh(surface.data)
    
    # Apply random displacement to vertices
    import random
    random.seed(42)  # For reproducible results
    
    for vert in bm.verts:
        # Add random displacement to Z coordinate
        displacement = random.uniform(-0.015, 0.015)  # watch out can stick throught bowl 
        vert.co.z += displacement
    
    # Update mesh
    bm.to_mesh(surface.data)
    bm.free()
    
    # Update the object
    surface.data.update()
    
    # Add material for better appearance
    material = bpy.data.materials.new(name="SurfaceMaterial")
    material.use_nodes = True
    
    # Clear default nodes
    material.node_tree.nodes.clear()
    
    # Create basic material nodes
    output_node = material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = material.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    
    # Set material properties with compatibility checks
    principled_node.inputs['Base Color'].default_value = (0.8, 0.8, 0.7, 1.0)  # Light gray
    principled_node.inputs['Roughness'].default_value = 0.8
    
    # Handle different Blender versions for specular/IOR inputs
    if 'Specular' in principled_node.inputs:
        principled_node.inputs['Specular'].default_value = 0.2
    elif 'Specular IOR' in principled_node.inputs:
        principled_node.inputs['Specular IOR'].default_value = 1.45
    elif 'IOR' in principled_node.inputs:
        principled_node.inputs['IOR'].default_value = 1.45
    
    # Connect nodes
    material.node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    
    # Assign material to surface
    if surface.data.materials:
        surface.data.materials[0] = material
    else:
        surface.data.materials.append(material)
    
    # Return BlenderProc mesh object using the mesh data, not the object
    return bproc.python.types.MeshObjectUtility.create_from_blender_mesh(surface.data)

def clear_scene_objects(exclude=None):
    """Clear scene objects except those in exclude list"""
    if exclude is None:
        exclude = []
    
    for obj in bpy.context.scene.objects:
        if obj.name not in exclude and obj.type == 'MESH':
            bpy.data.objects.remove(obj, do_unlink=True)

def position_object_on_surface(obj, surface):
    """Position object at origin and on top of surface"""
    # Get object's bounding box
    bbox_min = np.min([v for v in obj.get_bound_box()], axis=0)
    
    # Move object so its bottom is at z=0 (on surface)
    obj.set_location([0, 0, -bbox_min[2]])

def get_object_dimensions(obj):
    """Get object dimensions"""
    bbox = obj.get_bound_box()
    min_coords = np.min(bbox, axis=0)
    max_coords = np.max(bbox, axis=0)
    return max_coords - min_coords

def calculate_camera_distance(max_dimension, coverage_ratio):
    """Calculate camera distance to achieve desired object coverage, Minimal Focusing Distance important"""
    # Assuming 50mm lens (default), calculate distance for desired coverage
    # This is a simplified calculation
    fov_radians = np.radians(35)  # Approximate vertical FOV for 50mm lens
    distance = (max_dimension / coverage_ratio) / (2 * np.tan(fov_radians / 2))
    return max(distance, 2)  # Minimum distance of 2 units

def render_horizontal_cameras(obj, distance, num_cameras, obj_idx, phase_name, output_dir, append):
    """Render from horizontal cameras placed at the middle height of the object's bounding box focusing at its center."""

    # Get bounding box min and max in world coordinates
    bbox = obj.get_bound_box()  # Local-space bounding box corners
    l2w_mat = obj.get_local2world_mat()
    bbox_world = [l2w_mat @ mathutils.Vector((*corner, 1.0)) for corner in obj.get_bound_box()]


    # Compute min and max from world-space bbox
    bbox_min = mathutils.Vector((min([v[0] for v in bbox_world]),
                                 min([v[1] for v in bbox_world]),
                                 min([v[2] for v in bbox_world])))

    bbox_max = mathutils.Vector((max([v[0] for v in bbox_world]),
                                 max([v[1] for v in bbox_world]),
                                 max([v[2] for v in bbox_world])))

    # Compute the center point of the bounding box
    bbox_center = (bbox_min + bbox_max) / 2.0
    cam_height = bbox_center[2]
    target_location = bbox_center

    angles = np.linspace(0, 2 * np.pi, num_cameras, endpoint=False)

    # Place cameras in a horizontal circle around the object
    for angle in angles:
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        z = cam_height  # Camera height aligned with object's center height

        cam_location = mathutils.Vector((x, y, z))

        # Compute rotation to look at the bounding box center
        cam_rotation = bproc.camera.rotation_from_forward_vec(
            target_location - cam_location, inplane_rot=0.0
        )

        # Add camera pose
        bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(cam_location, cam_rotation))

    # Render all camera poses at once
    data = bproc.renderer.render()
    
    # Write BOP output for all frames (single call, positional args!)
    bproc.writer.write_bop(
        output_dir,
        [obj],
        data["depth"],   # positional
        data["colors"],  # positional
        annotation_unit='mm',
        append_to_existing_output=True
    )

    bproc.utility.reset_keyframes()



def render_top_cameras(obj, distance, num_cameras, angle_deg, obj_idx, phase_name, output_dir):
    """Render from top cameras at specified angle focusing on object's bbox center."""
    import numpy as np
    import mathutils
    import blenderproc as bproc

    # Compute bounding box center in world coordinates
    bbox = obj.get_bound_box()  # Local-space bounding box corners
    l2w_mat = obj.get_local2world_mat()
    bbox_world = [l2w_mat @ mathutils.Vector((*corner, 1.0)) for corner in bbox]

    bbox_min = mathutils.Vector((min([v[0] for v in bbox_world]),
                                 min([v[1] for v in bbox_world]),
                                 min([v[2] for v in bbox_world])))

    bbox_max = mathutils.Vector((max([v[0] for v in bbox_world]),
                                 max([v[1] for v in bbox_world]),
                                 max([v[2] for v in bbox_world])))

    bbox_center = (bbox_min + bbox_max) / 2.0

    # Camera intrinsics
    bproc.camera.set_intrinsics_from_blender_params(
        lens=50, lens_unit="MILLIMETERS", clip_start=0.1, clip_end=1000
    )
    
    angles = np.linspace(0, 2 * np.pi, num_cameras, endpoint=False)
    angle_rad = np.radians(angle_deg)
    
    for cam_idx, azimuth in enumerate(angles):
        # Compute horizontal radius and vertical elevation
        horizontal_distance = distance * np.cos(angle_rad)
        height = distance * np.sin(angle_rad)
        
        # Camera position in world coordinates
        x = bbox_center[0] + horizontal_distance * np.cos(azimuth)
        y = bbox_center[1] + horizontal_distance * np.sin(azimuth)
        z = bbox_center[2] + height
        
        cam_location = mathutils.Vector((x, y, z))
        target_location = bbox_center  # Focus on bbox center

        # Compute rotation to look at the target point
        cam_rotation = bproc.camera.rotation_from_forward_vec(
            target_location - cam_location, inplane_rot=0.0
        )

        # Add camera pose
        bproc.camera.add_camera_pose(
            bproc.math.build_transformation_mat(cam_location, cam_rotation)
        )
        
    # Render all camera poses at once
    data = bproc.renderer.render()
    
    # Write BOP output for all frames (single call, positional args!)
    bproc.writer.write_bop(
        output_dir,
        [obj],
        data["depth"],   # positional
        data["colors"],  # positional
        annotation_unit='mm',
        append_to_existing_output=True
    )

    bproc.utility.reset_keyframes()


def render_bottom_cameras(obj, distance, num_cameras, angle_deg, obj_idx, phase_name, output_dir):
    """Render from bottom cameras at specified angle (looking up) focused on bbox center."""
    import numpy as np
    import mathutils
    import blenderproc as bproc

    # Compute bounding box center in world coordinates
    bbox = obj.get_bound_box()
    l2w_mat = obj.get_local2world_mat()
    bbox_world = [l2w_mat @ mathutils.Vector((*corner, 1.0)) for corner in bbox]

    bbox_min = mathutils.Vector((min([v[0] for v in bbox_world]),
                                 min([v[1] for v in bbox_world]),
                                 min([v[2] for v in bbox_world])))

    bbox_max = mathutils.Vector((max([v[0] for v in bbox_world]),
                                 max([v[1] for v in bbox_world]),
                                 max([v[2] for v in bbox_world])))

    bbox_center = (bbox_min + bbox_max) / 2.0

    angles = np.linspace(0, 2 * np.pi, num_cameras, endpoint=False)
    angle_rad = np.radians(angle_deg)

    for cam_idx, azimuth in enumerate(angles):
        # Calculate camera position (below and angled up)
        horizontal_distance = distance * np.cos(angle_rad)
        depth = -distance * np.sin(angle_rad)  # Negative for below

        x = bbox_center[0] + horizontal_distance * np.cos(azimuth)
        y = bbox_center[1] + horizontal_distance * np.sin(azimuth)
        z = bbox_center[2] + depth  # Below the bbox center

        cam_location = mathutils.Vector((x, y, z))
        target_location = bbox_center  # Look at bbox center

        # Calculate rotation matrix to look upwards at target
        cam_rotation = bproc.camera.rotation_from_forward_vec(
            target_location - cam_location, inplane_rot=0.0
        )

        bproc.camera.add_camera_pose(
            bproc.math.build_transformation_mat(cam_location, cam_rotation)
        )

    # Render all camera poses at once
    data = bproc.renderer.render()

    # Write BOP output for all frames (single call, positional args!)
    bproc.writer.write_bop(
        output_dir,
        [obj],
        data["depth"],   # positional
        data["colors"],  # positional
        annotation_unit='mm',
        append_to_existing_output=True
    )
    
    bproc.utility.reset_keyframes()
    
def make_surface_transparent(surface):
    """Sets the surface material to be fully transparent"""
    material = surface.blender_obj.active_material
    if material is None:
        print("Surface has no material!")
        return
    
    # Use nodes
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    
    # Clear all nodes
    nodes.clear()
    
    # Add Transparent BSDF
    transparent_node = nodes.new(type='ShaderNodeBsdfTransparent')
    
    # Add Material Output node
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    
    # Link Transparent BSDF to Material Output
    links.new(transparent_node.outputs['BSDF'], output_node.inputs['Surface'])
    
    # Enable transparency in material settings
    material.blend_method = 'BLEND'
    material.shadow_method = 'NONE'  # Disable shadows from surface
    
def create_textured_surface(texture_path="/home/st3fan/Downloads/Plastic010_4K-JPG/"):
    """Create a structured surface (Oberfläche) with plastic texture from files
    
    Args:
        texture_path (str): Path to the directory containing texture files
    """
    
    # Create a plane
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, -0.02))
    surface = bpy.context.active_object
    surface.name = "surface"
    
    # Add subdivision for displacement detail
    bpy.context.view_layer.objects.active = surface
    bpy.ops.object.mode_set(mode='EDIT')
    #bpy.ops.mesh.subdivide(number_cuts=20)  # More subdivisions for displacement
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Create material with plastic texture maps
    material = bpy.data.materials.new(name="PlasticMaterial")
    material.use_nodes = True
    material.node_tree.nodes.clear()
    
    # Create shader nodes
    output_node = material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = material.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    
    # Ensure path ends with slash
    if not texture_path.endswith('/'):
        texture_path += '/'
    
    # Load and connect color texture
    color_file = texture_path + "Plastic010_4K-JPG_Color.jpg"
    if os.path.exists(color_file):
        color_tex = material.node_tree.nodes.new(type='ShaderNodeTexImage')
        color_img = bpy.data.images.load(color_file)
        color_tex.image = color_img
        material.node_tree.links.new(color_tex.outputs['Color'], principled_node.inputs['Base Color'])
    
    # Load and connect roughness texture
    rough_file = texture_path + "Plastic010_4K-JPG_Roughness.jpg"
    if os.path.exists(rough_file):
        rough_tex = material.node_tree.nodes.new(type='ShaderNodeTexImage')
        rough_img = bpy.data.images.load(rough_file)
        rough_tex.image = rough_img
        rough_tex.image.colorspace_settings.name = 'Non-Color'
        material.node_tree.links.new(rough_tex.outputs['Color'], principled_node.inputs['Roughness'])
    
    # Load and connect normal map
    normal_file = texture_path + "Plastic010_4K-JPG_NormalGL.jpg"
    if os.path.exists(normal_file):
        normal_tex = material.node_tree.nodes.new(type='ShaderNodeTexImage')
        normal_map = material.node_tree.nodes.new(type='ShaderNodeNormalMap')
        normal_img = bpy.data.images.load(normal_file)
        normal_tex.image = normal_img
        normal_tex.image.colorspace_settings.name = 'Non-Color'
        material.node_tree.links.new(normal_tex.outputs['Color'], normal_map.inputs['Color'])
        material.node_tree.links.new(normal_map.outputs['Normal'], principled_node.inputs['Normal'])
    
    # Add displacement modifier with displacement texture
    disp_file = texture_path + "Plastic010_4K-JPG_Displacement.jpg"
    if os.path.exists(disp_file):
        # Add subdivision surface modifier first
        subsurf_mod = surface.modifiers.new(name="Subdivision", type='SUBSURF')
        subsurf_mod.levels = 0
        
        # Add displacement modifier
        disp_mod = surface.modifiers.new(name="Displacement", type='DISPLACE')
        
        # Create displacement texture
        disp_texture = bpy.data.textures.new("DisplacementTexture", type='IMAGE')
        disp_img = bpy.data.images.load(disp_file)
        disp_texture.image = disp_img
        
        # Assign texture to modifier
        disp_mod.texture = disp_texture
        disp_mod.strength = 0.05  # Adjust displacement strength as needed
        disp_mod.mid_level = 0.5
    
    # Set material properties
    if 'Specular' in principled_node.inputs:
        principled_node.inputs['Specular'].default_value = 0.5
    elif 'Specular IOR' in principled_node.inputs:
        principled_node.inputs['Specular IOR'].default_value = 1.45
    elif 'IOR' in principled_node.inputs:
        principled_node.inputs['IOR'].default_value = 1.45
    
    # Connect principled BSDF to output
    material.node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    
    # Assign material to surface
    if surface.data.materials:
        surface.data.materials[0] = material
    else:
        surface.data.materials.append(material)
    
    # Add UV mapping coordinates
    bpy.context.view_layer.objects.active = surface
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.unwrap()
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Return BlenderProc mesh object using the mesh data, not the object
    return bproc.python.types.MeshObjectUtility.create_from_blender_mesh(surface.data)

if __name__ == "__main__":
    main()
