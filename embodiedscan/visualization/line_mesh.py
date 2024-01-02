"""Adapted from https://github.com/isl-
org/Open3D/pull/738#issuecomment-564785941.

Module which creates mesh lines from a line set
Open3D relies upon using glLineWidth to set line width on a LineSet
However, this method is now deprecated and not fully supporeted in
newer OpenGL versions.
See:
    Open3D Github Pull Request:
        https://github.com/intel-isl/Open3D/pull/738
    Other Framework Issues:
        https://github.com/openframeworks/openFrameworks/issues/3460

This module aims to solve this by converting a line into a triangular mesh
(which has thickness). The basic idea is to create a cylinder for each line
segment, translate it, and then rotate it.

License: MIT
"""
import numpy as np
import open3d as o3d


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """Aligns vector a to vector b with axis angle rotation."""
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):

    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular
        meshes.

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting
                line segments. If None, implicit lines from ordered pairwise
                points. (default: {None})
            colors {list} -- list of colors, or single color of the line
                (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(
                self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + \
                line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(translation,
                                                          relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(
                        axis_a))  # center=True)
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True,
                #   type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 \
                else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer."""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer."""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


if __name__ == '__main__':

    def main():
        print('Demonstrating LineMesh vs LineSet')
        # Create Line Set
        points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1],
                  [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7],
                 [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
        colors = [[1, 0, 0] for i in range(len(lines))]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # Create Line Mesh 1
        points = np.array(points) + [0, 0, 2]
        line_mesh1 = LineMesh(points, lines, colors, radius=0.02)
        line_mesh1_geoms = line_mesh1.cylinder_segments

        # Create Line Mesh 1
        points = np.array(points) + [0, 2, 0]
        line_mesh2 = LineMesh(points, radius=0.03)
        line_mesh2_geoms = line_mesh2.cylinder_segments

        o3d.visualization.draw_geometries(
            [line_set, *line_mesh1_geoms, *line_mesh2_geoms])

    main()
