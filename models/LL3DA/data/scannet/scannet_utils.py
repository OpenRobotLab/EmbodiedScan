"""Modified from: https://github.com/facebookresearch/votenet/blob/master/scann
et/scannet_utils.py."""

import csv
import json
import os
import sys

import numpy as np

try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print('pip install plyfile')
    sys.exit(-1)


def normalize_v3(arr):
    """Normalize a numpy array of 3 component vectors shape=(n,3)"""
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    arr[:, 0] /= (lens + 1e-8)
    arr[:, 1] /= (lens + 1e-8)
    arr[:, 2] /= (lens + 1e-8)
    return arr


def compute_normal(vertices, faces):
    #Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    normals = np.zeros(vertices.shape, dtype=vertices.dtype)
    #Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    normals[faces[:, 0]] += n
    normals[faces[:, 1]] += n
    normals[faces[:, 2]] += n
    normalize_v3(normals)

    return normals


def represents_int(s):
    """if string s represents an int."""
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename,
                       label_from='raw_category',
                       label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping


def read_mesh_vertices(filename):
    """read XYZ for each vertex."""
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
    return vertices


def read_mesh_vertices_rgb(filename):
    """read XYZ RGB for each vertex.

    Note: RGB values are in 0-255
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
        vertices[:, 3] = plydata['vertex'].data['red']
        vertices[:, 4] = plydata['vertex'].data['green']
        vertices[:, 5] = plydata['vertex'].data['blue']
    return vertices


def read_mesh_vertices_rgb_normal(filename):
    """read XYZ RGB normals point cloud from filename PLY file."""
    assert (os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 9], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
        vertices[:, 3] = plydata['vertex'].data['red']
        vertices[:, 4] = plydata['vertex'].data['green']
        vertices[:, 5] = plydata['vertex'].data['blue']

        # compute normals
        xyz = np.array([[x, y, z]
                        for x, y, z, _, _, _, _ in plydata['vertex'].data])
        face = np.array([f[0] for f in plydata['face'].data])
        nxnynz = compute_normal(xyz, face)
        vertices[:, 6:] = nxnynz
    return vertices
