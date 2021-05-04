"""
This module implements batched MeshCNN's feature extraction with PyTorch for acceleration purpose
"""
import torch
import numpy as np
from meshcnn.models.layers.mesh import Mesh
from meshcnn.models.layers.mesh_prepare import get_edge_points


def fixed_division(to_div, epsilon):
    if epsilon == 0:
        to_div[to_div == 0] = 0.1
    else:
        to_div += epsilon
    return to_div


def get_normals(vs, edge_points, side):
    edge_a = vs[:, edge_points[:, side // 2 + 2]] - vs[:, edge_points[:, side // 2]]
    edge_b = vs[:, edge_points[:, 1 - side // 2]] - vs[:, edge_points[:, side // 2]]
    normals = torch.cross(edge_a, edge_b)
    div = fixed_division(torch.norm(normals, dim=-1, keepdim=True), epsilon=0.1)
    normals /= div
    return normals


def get_opposite_angles(vs, edge_points, side):
    edges_a = vs[:, edge_points[:, side // 2]] - vs[:, edge_points[:, side // 2 + 2]]
    edges_b = vs[:, edge_points[:, 1 - side // 2]] - vs[:, edge_points[:, side // 2 + 2]]

    edges_a /= fixed_division(torch.norm(edges_a, dim=-1, keepdim=True), epsilon=0.1)
    edges_b /= fixed_division(torch.norm(edges_b, dim=-1, keepdim=True), epsilon=0.1)
    dot = torch.sum(edges_a * edges_b, dim=-1).clip(-1, 1)
    return torch.arccos(dot)


def get_ratios(vs, edge_points, side):
    edges_lengths = torch.norm(vs[:, edge_points[:, side // 2]] - vs[:, edge_points[:, 1 - side // 2]], dim=-1)

    point_o = vs[:, edge_points[:, side // 2 + 2]]
    point_a = vs[:, edge_points[:, side // 2]]
    point_b = vs[:, edge_points[:, 1 - side // 2]]
    line_ab = point_b - point_a
    projection_length = torch.sum(line_ab * (point_o - point_a), dim=-1) / fixed_division(
        torch.norm(line_ab, dim=-1), epsilon=0.1)
    closest_point = point_a + (projection_length / edges_lengths)[..., None] * line_ab
    d = torch.norm(point_o - closest_point, dim=-1)
    return d / edges_lengths


def dihedral_angle(mesh, edge_points):
    normals_a = get_normals(mesh, edge_points, 0)
    normals_b = get_normals(mesh, edge_points, 3)
    dot = torch.sum(normals_a * normals_b, dim=-1).clip(-1, 1)
    angles = np.pi - torch.arccos(dot)
    return angles.unsqueeze(1)


def symmetric_opposite_angles(vs, edge_points):
    """ computes two angles: one for each face shared between the edge
        the angle is in each face opposite the edge
        sort handles order ambiguity
    """
    angles_a = get_opposite_angles(vs, edge_points, 0).unsqueeze(1)
    angles_b = get_opposite_angles(vs, edge_points, 3).unsqueeze(1)
    angles = torch.cat((angles_a, angles_b), dim=1)
    angles = torch.sort(angles, dim=1)[0]
    return angles


def symmetric_ratios(vs, edge_points):
    """ computes two ratios: one for each face shared between the edge
        the ratio is between the height / base (edge) of each triangle
        sort handles order ambiguity
    """
    ratios_a = get_ratios(vs, edge_points, 0).unsqueeze(1)
    ratios_b = get_ratios(vs, edge_points, 3).unsqueeze(1)
    ratios = torch.cat((ratios_a, ratios_b), dim=1)
    ratios = torch.sort(ratios, dim=1)[0]
    return ratios


class FeatureExtractor:
    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.edge_points = get_edge_points(mesh)

    def extract_features(self, vs):
        features = []
        for extractor in [dihedral_angle, symmetric_opposite_angles, symmetric_ratios]:
            feature = extractor(vs, self.edge_points)
            features.append(feature)
        return torch.cat(features, dim=1)
