#USAGE : blender -b -P nbs_fbx_output.py -- --input ../demo --output ../demo/output.fbx (Default : Neural-BlendShape Model )

#USAGE : blender -b -P nbs_fbx_output.py -- --input ../demo --output ../demo/output.fbx --envelope_only 1 (Envelope-only Model)


import bpy

import numpy as np
from mathutils import Matrix,Vector,Quaternion

import os
import sys
import argparse





class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:] # the list after '--'
        except ValueError as e: # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())

def getKeyframes(ob):

    if ob.type in ['MESH','ARMATURE'] and ob.animation_data:
        for fc in ob.animation_data.action.fcurves :
            if fc.data_path.endswith('rotation_euler'):
                
                keyframe_list = []
                for key in fc.keyframe_points :
                    #print('frame:',key.co[0],'value:',key.co[1])
                    keyframe_list.append(key.co[0])

                keyframe_list = list(set(keyframe_list))
                firstKFN = int(keyframe_list[0])
                lastKFN = int(keyframe_list[-1])
                #Only needs to check animation of one bone
                return firstKFN, lastKFN


def init_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
def import_obj(filepath):
    bpy.ops.import_scene.obj(filepath=filepath,split_mode="OFF")
    bpy.ops.object.shade_smooth()
    # For some mysterious raison, this is necessary otherwise I cannot toggle shade smooth / shade flat

def import_skeleton(filepath):
    bpy.ops.import_anim.bvh(filepath=filepath, filter_glob="*.bvh", target='ARMATURE', global_scale=1, frame_start=1, use_fps_scale=False, use_cyclic=False, rotate_mode='NATIVE', axis_forward='-Z', axis_up='Y')



def export_animated_mesh(output_path,armature,mesh,IsAnimation):
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Select only skinned mesh and rig
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    mesh.select_set(True)

    if output_path.endswith('.glb'):
        print('Exporting to glTF binary (.glb)')
        # Currently exporting without shape/pose shapes for smaller file sizes
        bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB', export_selected=True, export_morph=IsAnimation)
    elif output_path.endswith('.fbx'):
        print('Exporting to FBX binary (.fbx)')
        bpy.ops.export_scene.fbx(filepath=output_path, use_selection=True, add_leaf_bones=False,bake_anim=IsAnimation)


    else:
        print('ERROR: Unsupported export format: ' + output_path)
        sys.exit(1)

    return




if __name__ == '__main__':
    try:
        if bpy.app.background:

            parser = ArgumentParserForBlender()
            #parser = argparse.ArgumentParser(description='Create keyframed animated skinned SMPL mesh from VIBE output')
            parser.add_argument('--input', dest='input_dir', type=str, required=True,
                                help='Input directory')
            parser.add_argument('--output', dest='output_path', type=str, required=True,
                                help='Output file or directory')

            parser.add_argument('--envelope_only', dest='envelope_only', type=int, default=0,
                                help='set envelope_only')
            
            args = parser.parse_args()  
            
            input_dir = args.input_dir
            output_path = args.output_path
            IsEnvelope = args.envelope_only
            
            obj_path = os.path.join(input_dir,'T-pose.obj')
            skeleton_path = os.path.join(input_dir,'skeleton.bvh')
            weight_path = os.path.join(input_dir,'weight.npy')
            basis_path = os.path.join(input_dir,'basis.npy')
            coff_path = os.path.join(input_dir,'coff.npy')

            


            init_scene()

            import_obj(obj_path)
            mesh = bpy.context.selected_objects[0]

            import_skeleton(skeleton_path)
            skeleton = bpy.context.selected_objects[0]

            firstKFN, lastKFN  = getKeyframes(skeleton)
            IsAnimation = not(firstKFN==lastKFN)


            mesh.select_set(False)
            skeleton.select_set(True)
            bpy.context.view_layer.objects.active = skeleton

            bpy.ops.object.mode_set(mode='POSE')

            #Clear the transform to combine rig with T-pose
            bpy.ops.pose.transforms_clear()


            bpy.ops.object.mode_set(mode='OBJECT')



            mesh.select_set(True)
            skeleton.select_set(True)
            bpy.context.view_layer.objects.active = skeleton

            bpy.ops.object.parent_set(type='ARMATURE_AUTO')
                 

            weight = np.load(weight_path)


            for vg in range(weight.shape[1]):
                vg_index = str(vg).zfill(2)
                for i in range(weight[:,vg].shape[0]):
                    if weight[:,vg][i]> 0:
                        mesh.vertex_groups[vg_index].add([i], weight[:,vg][i], 'REPLACE')

            if IsAnimation:
                #Set Frame start and end
                bpy.data.scenes[0].frame_start = firstKFN
                bpy.data.scenes[0].frame_end = lastKFN
                print("This is Animated Model")

                # Check if basis and coff npy files exist
                # basis and coff npy files are generated from params (basis_full, coff) in ./architecture/blend_shapes.py

                if not (os.path.exists(basis_path) and os.path.exists(coff_path)):
                    print("basis and coff file do not exist")
                    IsEnvelope = 1

                if IsEnvelope!=1:
                    print("Neural-BlendShape Model is generated")
                    basis = np.load(basis_path)
                    coff = np.load(coff_path)


                    #print(basis.shape)
                    #print(coff.shape)


                    verts = mesh.data.vertices

                    #clear shape keys
                    mesh.shape_key_clear()

                    sk_basis = mesh.shape_key_add(from_mix=False)




                    frame_num = coff.shape[0]


                    for n in range(basis.shape[1]):
                        # Create new shape key
                        key_name = str(n).zfill(3)
                        
                        sk = mesh.shape_key_add(name=key_name,from_mix=False)
                        sk.interpolation = 'KEY_LINEAR'

                        # position each vert
                        for i in range(len(verts)):
                            sk.data[i].co.x += basis[i,n,0]
                            sk.data[i].co.y += basis[i,n,1]
                            sk.data[i].co.z += basis[i,n,2]
                            
                        sk = mesh.shape_key_add(name=key_name+'_neg',from_mix=False)
                        sk.interpolation = 'KEY_LINEAR'

                        # position each vert
                        for i in range(len(verts)):
                            sk.data[i].co.x -= basis[i,n,0]
                            sk.data[i].co.y -= basis[i,n,1]
                            sk.data[i].co.z -= basis[i,n,2]


                    for n in range(frame_num):
                        for i in range(coff.shape[1]):

                            if coff[n][i]>=0:
                                key_name = str(i).zfill(3)
                                mesh.data.shape_keys.key_blocks[key_name].value = coff[n][i]
                                mesh.data.shape_keys.key_blocks[key_name].keyframe_insert(data_path='value', frame=n+1)
                                key_name = str(i).zfill(3) +'_neg'
                                mesh.data.shape_keys.key_blocks[key_name].value = 0
                                mesh.data.shape_keys.key_blocks[key_name].keyframe_insert(data_path='value', frame=n+1)
                            else :
                                key_name = str(i).zfill(3)
                                mesh.data.shape_keys.key_blocks[key_name].value = 0
                                mesh.data.shape_keys.key_blocks[key_name].keyframe_insert(data_path='value', frame=n+1)
                                key_name = str(i).zfill(3) +'_neg'
                                mesh.data.shape_keys.key_blocks[key_name].value = -1*coff[n][i]
                                mesh.data.shape_keys.key_blocks[key_name].keyframe_insert(data_path='value', frame=n+1)
                else:
                    print("Envelope-only Model is generated")
            else :
                print("This is Static Model")
            export_animated_mesh(output_path,skeleton,mesh,IsAnimation) 
    except SystemExit as ex:
        if ex.code is None:
            exit_status = 0
        else:
            exit_status = ex.code

        print('Exiting. Exit status: ' + str(exit_status))

        # Only exit to OS when we are not running in Blender GUI
        if bpy.app.background:
            sys.exit(exit_status)