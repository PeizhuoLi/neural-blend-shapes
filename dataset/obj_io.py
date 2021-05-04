import numpy as np


def load_obj(filename):
    """
    This code is adapted from MeshCNN
    """
    vs, faces = [], []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            split_line = line.split()
            if not split_line:
                continue
            elif split_line[0] == 'v':
                vs.append([float(v) for v in split_line[1:4]])
            elif split_line[0] == 'f':
                face_vertex_ids = [int(c.split('/')[0]) for c in split_line[1:]]
                assert len(face_vertex_ids) == 3
                face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                                   for ind in face_vertex_ids]
                faces.append(face_vertex_ids)
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=np.int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return vs, faces


def write_obj(filename, vs, faces):
    faces = faces + 1
    with open(filename, 'w') as f:
        for vi, v in enumerate(vs):
            f.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for i, face in enumerate(faces):
            f.write('f %d %d %d\n' % (face[0], face[1], face[2]))
