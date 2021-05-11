import sys
import bpy
import numpy as np
import random
import argparse


def generate_color_table(num_colors):
    """
    Generate num_colors of colors from matplotlib's tableau colors (saved as npy), repeat if necessary
    @param num_colors: int, number of colors needed
    @return: rgb, (num_colors, 3)
    """
    base_colors = np.load('./tableau_color.npy')
    idx = list(range(base_colors.shape[0]))
    random.seed(5)
    random.shuffle(idx)
    pt = 0
    res = []
    for i in range(num_colors):
        res.append(base_colors[idx[pt]])
        pt += 1
        pt %= base_colors.shape[0]
    return np.array(res)


def weight2color(weight):
    n_color = weight.shape[1]
    colors = generate_color_table(n_color)
    res = np.matmul(weight, colors)
    return res


def add_material_for_mesh(objs):
    mat = bpy.data.materials.new(name='VertexColor')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    vert_color = mat.node_tree.nodes.new(type="ShaderNodeVertexColor")

    bsdf.inputs[5].default_value = 0

    links = mat.node_tree.links
    links.new(vert_color.outputs[0], bsdf.inputs[0])

    for obj in objs:
        if len(obj.data.materials):
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)


def load_vert_col(me, colors):
    vcols = me.data.vertex_colors
    polys = me.data.polygons

    vcol = vcols.new(name="Visualization")

    idx = 0
    for poly in polys:
        verts = poly.vertices
        for i, _ in enumerate(poly.loop_indices):
            c = colors[verts[i]]
            vcol.data[idx].color = (c[0], c[1], c[2], 1.0)
            idx += 1


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_path', type=str, default='../demo/T-pose.obj')
    parser.add_argument('--weight_path', type=str, default='../demo/weight.npy')
    return parser


def change_shading_mode(shading_mode):
    """
    https://blender.stackexchange.com/questions/124347/blender-2-8-python-code-to-switch-shading-mode-between-wireframe-and-solid-mo/124427
    """
    for area in bpy.context.workspace.screens[0].areas:
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = shading_mode


if __name__ == '__main__':
    parser = get_parser()
    argv = sys.argv
    if "--" not in argv:
        argv = []  # as if no args are passed
    else:
        argv = argv[argv.index("--") + 1:]
    args = parser.parse_args(argv)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    weight = np.load(args.weight_path)
    color = weight2color(weight)

    bpy.ops.import_scene.obj(filepath=args.obj_path, split_mode='OFF')
    me = bpy.context.selected_objects[0]
    bpy.ops.object.shade_smooth()

    load_vert_col(me, color)
    add_material_for_mesh([me])

    change_shading_mode('MATERIAL')
