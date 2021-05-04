import sys
import bpy
import os
import numpy as np
import argparse


def mark_all_edges_freestyle(mesh):
    print(mesh.name_full)
    for edge in mesh.data.edges:
        edge.use_freestyle_mark = True


def load_obj(source):
    old_objs = set(bpy.context.scene.objects)
    bpy.ops.import_scene.obj(filepath=source, split_mode='OFF')
    imported_objs = set(bpy.context.scene.objects) - old_objs
    return list(imported_objs)


def add_material_for_objs(objs, name):
    for obj in objs:
        obj.data.materials.clear()
        for mat in bpy.data.materials:
            if name == mat.name:
                obj.data.materials.append(mat)
                return
    raise Exception("This line shouldn't be reached")


def render_obj(filename, extra_pos=None):
    obj = load_obj(filename)[0]
    add_material_for_objs([obj], 'character')
    if use_free_style:
        mark_all_edges_freestyle(obj)
    if extra_pos is not None:
        obj.location = extra_pos
    bpy.ops.object.shade_smooth()

    bpy.ops.render.render(use_viewport=True, animation=True)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.delete()


def batch_render(dir, extra_pos=None):
    files = [f for f in os.listdir(dir) if f.endswith('.obj')]
    if extra_pos is not None:
        extra_pos = np.load(extra_pos)
    files.sort()
    for i, file in enumerate(files):
        bpy.context.scene.frame_start = i
        bpy.context.scene.frame_end = i
        bpy.context.scene.frame_current = i
        if extra_pos is None:
            render_obj(os.path.join(dir, file))
        else:
            render_obj(os.path.join(dir, file), extra_pos[i])


def aggregate_video(image_path, video_path):
    cmd = 'ffmpeg -r 30 -i %s%%04d.png -vcodec qtrle %s' % (image_path, video_path)
    cmd = 'ffmpeg -r 30 -i %s%%04d.png -pix_fmt rgba -c:v png -c:a copy %s' % (image_path, video_path)
    print(cmd)
    os.system(cmd)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='../demo/images/')
    parser.add_argument('--video_path', type=str, default='../demo/video.mov')
    parser.add_argument('--obj_path', type=str, default='../demo/obj')
    parser.add_argument('--render_mesh_connectivity', type=int, default=0)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    argv = sys.argv
    if "--" not in argv:
        argv = []  # as if no args are passed
    else:
        argv = argv[argv.index("--") + 1:]
    args = parser.parse_args(argv)

    images_path = args.image_path
    video_path = args.video_path

    use_free_style = bool(args.render_mesh_connectivity)

    bpy.context.scene.render.use_freestyle = use_free_style

    os.system(f"rm {os.path.join(images_path, '*.png')}")
    bpy.context.scene.render.filepath = images_path

    batch_render(args.obj_path)

    aggregate_video(images_path, video_path)
