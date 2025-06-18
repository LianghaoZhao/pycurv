import vtk
import numpy as np
import time
from scipy import ndimage
import math
from graph_tool import GraphView, incident_edges_op
from graph_tool.topology import (shortest_distance, label_largest_component,
                                 label_components)

from . import graphs
from . import pexceptions
from .pycurv_io import TypesConverter
from .curvature_definitions import *
from .surface import (add_point_normals_to_vtk_surface,
                      add_curvature_to_vtk_surface, rescale_surface)
from .linalg import (
    perpendicular_vector, rotation_matrix, rotate_vector, signum, nice_acos,
    triangle_normal, triangle_center, triangle_area_cross_product)

"""
Set of functions and classes (abstract SurfaceGraph and derived TriangleGraph)
for representing a surface by a graph, cleaning the surface and triangle-wise
operations of the normal vector voting algorithm.

Author: Maria Salfer (Max Planck Institute for Biochemistry)
"""

__author__ = 'Maria Salfer'


class SurfaceGraph(graphs.SegmentationGraph):
    """Class defining the abstract SurfaceGraph object."""

    def build_graph_from_vtk_surface(self, surface, scale, verbose=False):
        """
        Base method for building a graph from a vtkPolyData surface, to be
        implemented by SurfaceGraph subclasses.

        Args:
            surface (vtk.vtkPolyData): a signed surface (mesh of triangles)
                generated from the segmentation in voxels
            scale (tuple): pixel size (X, Y, Z) in given units for scaling the
                surface and the graph
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out

        Returns:
            None
        """
        pass

    # * The following SurfaceGraph method is implementing with adaptations
    # the second part of the first step of normal vector voting algorithm of
    # Page et al., 2002. *
    def estimate_normal(self, vertex_v_ind, V_v, epsilon=0, eta=0):
        """
        For a vertex v, its calculated matrix V_v (output of
        collect_normal_votes) and the parameters epsilon and eta (default 2
        each), classifies its orientation.

        The output classes are 1 if it belongs to a surface patch, 2 if it
        belongs to a crease junction or 3 if it doesn't have a preferred
        orientation.

        This is done using eigen-decomposition of V_v and equations (9) and (10)
        from the paper of Page et al., 2002. Equations (11) and (12) may help to
        choose epsilon and eta.

        Args:
            vertex_v_ind (int): index of the vertex v in the surface
                triangle-graph whose orientation is classified
            V_v (numpy.ndarray): the 3x3 symmetric matrix V_v
            epsilon (float, optional): parameter of Normal Vector Voting
                algorithm influencing the number of triangles classified as
                "crease junction" (class 2), default 0
            eta (float, optional): parameter of Normal Vector Voting algorithm
                influencing the number of triangles classified as "crease
                junction" (class 2) and "no preferred orientation" (class 3, see
                Notes), default 0

        Returns:
            orientation of vertex v (int): 1 if it belongs to a surface patch,
                2 if it belongs to a crease junction or 3 if it doesn't have a
                preferred orientation
            estimated normal "n_v" (3x1 array) if class is 1, otherwise zeros
            estimated tangent "t_v" (3x1 array) if class is 2, otherwise zeros

        Notes:
            If epsilon = 0 and eta = 0, all triangles will be classified as
            "surface patch" (class 1).
        """
        vertex_v = self.graph.vertex(vertex_v_ind)
        # Decompose the symmetric semidefinite matrix V_v:
        # eigenvalues are in increasing order and eigenvectors are in columns of
        # the returned quadratic matrix
        eigenvalues, eigenvectors = np.linalg.eigh(V_v)
        # Eigenvalues from highest to lowest:
        e_1_val = eigenvalues[2]
        e_2_val = eigenvalues[1]
        e_3_val = eigenvalues[0]
        # Eigenvectors, corresponding to the eigenvalues:
        e_1_vec = eigenvectors[:, 2]
        e_3_vec = eigenvectors[:, 0]
        # Saliency maps:
        S_s = e_1_val - e_2_val  # surface patch
        S_c = e_2_val - e_3_val  # crease junction
        S_n = e_3_val  # no preferred orientation

        # Make decision and add the estimated normal or tangent as properties to
        # the graph (add a placeholder [0, 0, 0] to each property, where it does
        # not apply):
        max_saliency = max(S_s, epsilon * S_c, epsilon * eta * S_n)
        if max_saliency == S_s:
            # Eventually have to flip (negate) the estimated normal, because its
            # direction is lost during the matrix generation!
            # Take the one for which the angle to the original normal is smaller
            # (or cosine of the angle is higher):
            normal1 = e_1_vec
            normal2 = -e_1_vec
            orig_normal = self.graph.vp.normal[vertex_v]
            cos_angle1 = np.dot(orig_normal, normal1)
            cos_angle2 = np.dot(orig_normal, normal2)
            if cos_angle1 > cos_angle2:
                n_v = normal1
            else:
                n_v = normal2
            return 1, n_v, np.zeros(shape=3)
        elif max_saliency == (epsilon * S_c):
            return 2, np.zeros(shape=3), e_3_vec
        else:
            return 3, np.zeros(shape=3), np.zeros(shape=3)

    # * The following SurfaceGraph methods are implementing with adaptations
    # the second step of normal vector voting algorithm of Page et al., 2002. *
    def collect_curvature_votes(
            self, vertex_v_ind, g_max, sigma, full_dist_map=None,
            page_curvature_formula=False, a_max=0.0):
        """
        For a vertex v, collects the curvature and tangent votes of all
        triangles within its geodesic neighborhood belonging to a surface patch
        and calculates the matrix B_v.

        Implements equations (13) and (15) also illustrated in figure 6(c),
        (16), (17), (14), and (5) from the paper. In short, three components are
        calculated for each neighboring vertex (representing triangle centroid)
        v_i:

        1. Weight w_i depending on the geodesic distance between v_i and v (so
            that all weights sum up to 2 * pi).

        2. Tangent t_i from v in the direction of the arc connecting v and v_i
            (using the estimated normal n_v at v).

        3. Normal curvature kappa_i from Tong and Tang et al. defined as:
            kappa_i = abs(2 * cos((pi - phi) / 2) / vector_length(vv_i)),
            where phi is the turning angle between n_v and the projection
            n_v_i_p of the estimated normal of v_i (n_v_i) onto the arc plane
            (formed by v, n_v and v_i); sign of kappa_i is opposite to the one
            of dot(t_i, n_v_i_p)

        Those components are incorporated into the 3x3 symmetric matrix B_v
        (Eq. 5), which is returned.

        Args:
            vertex_v_ind (int): index of the vertex v in the surface
                triangle-graph for which the votes are collected
            g_max (float): the maximal geodesic distance in units of the graph
            sigma (float): sigma, defined as 3*sigma = g_max, so that votes
                beyond the neighborhood can be ignored
            full_dist_map (graph_tool.PropertyMap, optional): the full distance
                map for the whole graph; if None, a local distance map is
                calculated for this vertex (default)
            page_curvature_formula (boolean, optional): if True (default False)
                normal curvature definition from Page et al. is used:
                the turning angle phi between n_v and the projection n_v_i_p
                of the estimated normal of v_i (n_v_i) onto the arc plane
                (formed by v, n_v and v_i) divided by the arc length (geodesic
                distance between v and v_i).
            a_max (float, optional): if given (default 0.0), votes are
                weighted by triangle area like in the first pass (normals
                estimation)
        Returns:
            the 3x3 symmetric matrix B_v (numpy.ndarray)
        """
        vertex_v = self.graph.vertex(vertex_v_ind)

        # To spare function referencing every time in the following for loop:
        vertex = self.graph.vertex
        vp_n_v = self.graph.vp.n_v
        exp = math.exp
        xyz = self.graph.vp.xyz
        array = np.array
        dot = np.dot
        sqrt = math.sqrt
        pi = math.pi
        cross = np.cross
        acos = nice_acos
        outer = np.multiply.outer
        cos = math.cos
        if a_max > 0:
            area = self.graph.vp.area

        # Find the neighboring vertices of vertex v to be returned:
        if self.__class__.__name__ == "TriangleGraph":
            if vertex_v_ind == 0:
                print("Calling find_geodesic_neighbors")
            # leave only neighbors belonging to a surface patch
            neighbor_idx_to_dist = self.find_geodesic_neighbors(
                vertex_v, g_max, full_dist_map=full_dist_map, only_surface=True)
        else:  # PointGraph
            if vertex_v_ind == 0:
                print("Calling find_geodesic_neighbors_exact")
            # stop looking if neighbor is not in a surface patch
            neighbor_idx_to_dist = self.find_geodesic_neighbors_exact(
                vertex_v, g_max, verbose=False, only_surface=True)
        # Doing it again, because saving in first pass caused memory problems
        try:
            assert(len(neighbor_idx_to_dist) > 0)
        except AssertionError:
            print("{} neighbors in a surface patch with weights w_i:".format(
                len(neighbor_idx_to_dist)))
            print("The vertex will be ignored.")
            return None

        # Get the coordinates of vertex v and its estimated normal n_v (as numpy
        # array):
        v = array(xyz[vertex_v])
        n_v = array(vp_n_v[vertex_v])

        # Initialize the weighted matrix sum of all votes for vertex v to be
        # calculated:
        B_v = np.zeros(shape=(3, 3))

        # Store all the weights w_i of the neighboring triangles because
        # they will have to be normalized to sum up to 2 * pi:
        all_w_i = []

        # Let each of the neighboring triangles belonging to a surface patch
        # (as checked before) to cast a vote on vertex v:
        for i, idx_v_i in enumerate(neighbor_idx_to_dist.keys()):
            # Get the neighboring vertex v_i:
            vertex_v_i = vertex(idx_v_i)

            # First, calculate the weight depending on the geodesic distance to
            # the neighboring vertex v_i from the vertex v, g_i:
            g_i = neighbor_idx_to_dist[idx_v_i]
            w_i = exp(- g_i / sigma)
            if a_max > 0:
                a_i = area[vertex_v_i]
                w_i *= a_i / a_max
            all_w_i.append(w_i)

            # Second, calculate tangent directions t_i of each vote:
            v_i = array(xyz[vertex_v_i])
            vv_i = v_i - v
            t_i = vv_i - dot(n_v, vv_i) * n_v
            t_i_len = sqrt(dot(t_i, t_i))
            t_i = t_i / t_i_len

            # Third, calculate the normal curvature kappa_i:
            # p_i: vector perpendicular to the plane that contains both n_v and
            # t_i as well as the normal curve (between v and v_i)
            p_i = cross(n_v, t_i)
            n_v_i = array(vp_n_v[vertex_v_i])  # estimated normal of vertex v_i
            # n_v_i_p: projection of n_v_i on the plane containing n_v rooted at
            # v and v_i
            n_v_i_p = n_v_i - dot(p_i, n_v_i) * p_i
            n_v_i_p_len = sqrt(dot(n_v_i_p, n_v_i_p))
            # 0 <= phi <= pi is the turning angle between n_v_i_p and n_v
            cos_phi = float(dot(n_v, n_v_i_p) / n_v_i_p_len)
            phi = acos(cos_phi)
            if page_curvature_formula:  # formula from Page et al. paper:
                s = neighbor_idx_to_dist[idx_v_i]  # arc length s = g_i
                kappa_i = phi / s
                # decomposition does not work if multiply kappa_i with its sign
            else:  # formula from Tong and Tang paper:
                kappa_i = abs(2 * cos((pi - phi) / 2) / sqrt(dot(vv_i, vv_i)))
                # curvature sign has to be negated according to our surface
                # normals convention (point towards inside of a convex surface):
                kappa_i_sign = -1 * signum(dot(t_i, n_v_i_p))
                kappa_i *= kappa_i_sign

            # Finally, sum up the components of B_v:
            B_v += w_i * kappa_i * outer(t_i, t_i)

        # Calculate the factor the weights have to be multiplied with in order
        # to sum up to 2 * pi
        all_w_i = array(all_w_i)
        sum_w_i = np.sum(all_w_i)
        wanted_sum_w_i = 2 * pi
        factor = wanted_sum_w_i / sum_w_i

        # Normalize B_v by factor / (2 * pi):
        B_v *= factor / (2 * pi)
        return B_v

    def estimate_curvature(self, vertex_v_ind, B_v):
        """
        For a vertex v and its calculated matrix B_v (output of
        collect_curvature_votes), calculates the principal directions (t_1 and
        t_2) and curvatures (kappa_1 and kappa_2) at this vertex.

        This is done using eigen-decomposition of B_v: the eigenvectors
        corresponding to the two largest eigenvalues are the principal
        directions and the principal curvatures are found with linear
        transformations of those eigenvalues (Eq. 4).

        Args:
            vertex_v_ind (int): index of the vertex v in the surface
                triangle-graph for which the principal directions and curvatures
                are estimated
            B_v (numpy.ndarray): the 3x3 symmetric matrix B_v (output of
                collect_curvature_votes)

        Returns:
            t_1, t_2, kappa_1, kappa_2, gauss_curvature, mean_curvature,
            shape_index, curvedness
            if B_v is None or the decomposition does not work, a list of None
        """
        if B_v is None or math.isnan(B_v[0, 0]):
            return None, None, None, None, None, None, None, None

        vertex_v = self.graph.vertex(vertex_v_ind)

        # Decompose the symmetric matrix B_v:
        # eigenvalues are in increasing order and eigenvectors are in columns of
        # the returned quadratic matrix
        eigenvalues, eigenvectors = np.linalg.eigh(B_v)
        # Eigenvalues from highest to lowest:
        b_1 = eigenvalues[2]
        b_2 = eigenvalues[1]
        b_3 = eigenvalues[0]
        # Eigenvectors that correspond to the highest two eigenvalues are the
        # estimated principal directions:
        t_1 = eigenvectors[:, 2]
        t_2 = eigenvectors[:, 1]
        t_3 = eigenvectors[:, 0]  # has to be equal to the normal n_v or -n_v
        n_v = np.array(self.graph.vp.n_v[vertex_v])
        try:
            assert(round(abs(t_3[0]), 7) == round(abs(n_v[0]), 7) and
                   round(abs(t_3[1]), 7) == round(abs(n_v[1]), 7) and
                   round(abs(t_3[2]), 7) == round(abs(n_v[2]), 7))
        except AssertionError:
            if (round(abs(t_1[0]), 7) == round(abs(n_v[0]), 7) and
                    round(abs(t_1[1]), 7) == round(abs(n_v[1]), 7) and
                    round(abs(t_1[2]), 7) == round(abs(n_v[2]), 7)):
                t_1 = t_3  # t_3 = n_v
                b_1 = b_3  # b_3 = 0
            elif (round(abs(t_2[0]), 7) == round(abs(n_v[0]), 7) and
                    round(abs(t_2[1]), 7) == round(abs(n_v[1]), 7) and
                    round(abs(t_2[2]), 7) == round(abs(n_v[2]), 7)):
                t_2 = t_3  # t_3 = n_v
                b_2 = b_3  # b_3 = 0
            else:
                print("Error: no eigenvector which equals to the normal found")
                print("t_1 = {}".format(t_1))
                print("t_2 = {}".format(t_2))
                print("t_3 = {}".format(t_3))
                print("n_v = {}".format(n_v))
                print("b_1 = {}".format(b_1))
                print("b_2 = {}".format(b_2))
                print("b_3 = {}".format(b_3))
                return None, None, None, None, None, None, None, None
        # Estimated principal curvatures:
        kappa_1 = 3 * b_1 - b_2
        kappa_2 = 3 * b_2 - b_1
        # Curvatures and directions might be interchanged:
        if kappa_1 < kappa_2:
            t_1, t_2 = t_2, t_1
            kappa_1, kappa_2 = kappa_2, kappa_1

        # return t_1, t_2, kappa_1, kappa_2, Gaussian, mean curvature,
        # shape index and curvedness of vertex v:
        gauss_curvature = calculate_gauss_curvature(kappa_1, kappa_2)
        mean_curvature = calculate_mean_curvature(kappa_1, kappa_2)
        shape_index = calculate_shape_index(kappa_1, kappa_2)
        curvedness = calculate_curvedness(kappa_1, kappa_2)
        return (t_1, t_2, kappa_1, kappa_2,
                gauss_curvature, mean_curvature, shape_index, curvedness)

    def first_pass(self, vertex_v_ind, g_max, a_max, sigma, full_dist_map=None):
        """
        Combines the two steps of the first pass: normal votes collection
        and normals estimation. For more information, see the functions
        collect_normal_votes() (of TriangleGraph or PointGraph) and
        estimate_normal() (of TriangleGraph).

        Args:
            vertex_v_ind (int): index of the vertex v in the surface
                triangle-graph for which the votes are collected
            g_max (float): the maximal geodesic distance in units of the graph
            a_max (float): the area of the largest triangle in the surface
                triangle-graph
            sigma (float): sigma, defined as 3*sigma = g_max, so that votes
                beyond the neighborhood can be ignored
            full_dist_map (graph_tool.PropertyMap, optional): the full distance
                map for the whole graph; if None, a local distance map is
                calculated for this vertex (default)

        Returns:
            orientation of vertex v (int): 1 if it belongs to a surface patch,
                2 if it belongs to a crease junction or 3 if it doesn't have a
                preferred orientation
            estimated normal "n_v" (3x1 array) if class is 1, otherwise zeros
            estimated_tangent "t_v" (3x1 array) if class is 2, otherwise zeros
        """
        if self.__class__.__name__ == 'TriangleGraph':
            num_neighbors, V_v = self.collect_normal_votes(
                vertex_v_ind, g_max, a_max, sigma, full_dist_map)
        else:  # PointGraph
            num_neighbors, V_v = self.collect_normal_votes(
                vertex_v_ind, g_max, a_max, sigma)
        class_v, n_v, t_v = self.estimate_normal(
            vertex_v_ind, V_v, epsilon=0, eta=0)
        return num_neighbors, class_v, n_v, t_v

    def second_pass(self, vertex_v_ind, g_max, sigma, full_dist_map=None,
                    page_curvature_formula=False, a_max=0.0):
        """
        Combines the two steps of the second pass: curvature votes collection
        and curvature estimation. For more information, see SurfaceGraph
        functions collect_curvature_votes() and estimate_curvature().

        Args:
            vertex_v_ind (int): index of the vertex v in the surface
                triangle-graph for which the votes are collected
            g_max (float): the maximal geodesic distance in units of the graph
            sigma (float): sigma, defined as 3*sigma = g_max, so that votes
                beyond the neighborhood can be ignored
            full_dist_map (graph_tool.PropertyMap, optional): the full distance
                map for the whole graph; if None, a local distance map is
                calculated for this vertex (default)
            page_curvature_formula (boolean, optional): if True (default False)
                normal curvature definition from Page et al. is used:
                the turning angle phi between n_v and the projection n_v_i_p
                of the estimated normal of v_i (n_v_i) onto the arc plane
                (formed by v, n_v and v_i) divided by the arc length (geodesic
                distance between v and v_i).
            a_max (float, optional): if given (default 0.0), votes are
                weighted by triangle area like in the first pass (normals
                estimation)

        Returns:
            t_1, t_2, kappa_1, kappa_2, gauss_curvature, mean_curvature,
            shape_index, curvedness or Nones if vertex v does not have any
            neighbor belonging to a surface patch
        """
        B_v = self.collect_curvature_votes(
            vertex_v_ind, g_max, sigma, full_dist_map,
            page_curvature_formula, a_max)
        return self.estimate_curvature(vertex_v_ind, B_v)

    def gen_curv_vote(self, poly_surf, vertex_v, radius_hit):
        """
        Implements the third pass of the method of Tong & Tang et al., 2005,
        "Algorithm 5. GenCurvVote". Estimates principal curvatures and
        directions (after normals estimation) using curvature tensor. For the
        given triangle center, eight neighboring points are sampled around it at
        equal angles using tangents of length defined by the radius_hit
        parameter.

        Args:
            poly_surf (vtkPolyData): surface from which the graph was generated,
                scaled to given units
            vertex_v (graph_tool.Vertex): the vertex v in the surface
                triangle-graph for which the principal directions and curvatures
                are estimated
            radius_hit (float): radius in length unit of the graph for sampling
                surface points in tangent directions
        """
        # Get the coordinates of vertex v and its estimated normal n_v (as numpy
        # array, input of the original method):
        v = np.array(self.graph.vp.xyz[vertex_v])
        n_v = np.array(self.graph.vp.n_v[vertex_v])

        # Define a cellLocator to be able to compute intersections between lines
        # and the surface:
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(poly_surf)
        locator.BuildLocator()
        tolerance = 0.001

        # Define some frequently used functions in the loop:
        multiply = np.multiply
        sqrt = math.sqrt
        dot = np.dot
        outer = np.multiply.outer
        pi = math.pi

        # a vector on tangent plane T_r(S)
        votedir = perpendicular_vector(n_v)
        if votedir is None:
            print("Error: calculation of a perpendicular vector failed")
            exit(0)
        # B_v = np.zeros(shape=(2, 2))  # when transform votedir to 2D
        B_v = np.zeros(shape=(3, 3))
        R = rotation_matrix(n_v, pi/4)
        for i in range(8):
            # rotate the vector by 45 degrees (pi/4 radians) around n_v axis
            votedir = rotate_vector(votedir, pi/4, matrix=R)
            v_t = v + votedir * radius_hit

            # Find intersection point c between the surface and line segment l
            # going through v_t and parallel to n_v:
            # point on line l from v_t in normal direction
            p1 = v_t + multiply(n_v, radius_hit)
            # point on line l from v_t in opposite normal direction
            p2 = v_t - multiply(n_v, radius_hit)
            # Outputs (we need only c, which is the x, y, z position
            # of the intersection):
            t = vtk.mutable(0)
            c = [0.0, 0.0, 0.0]
            pcoords = [0.0, 0.0, 0.0]
            sub_id = vtk.mutable(0)
            cell_id = vtk.mutable(0)
            locator.IntersectWithLine(p1, p2, tolerance, t, c, pcoords,
                                      sub_id, cell_id)
            # If there is no intersection, c stays like initialized:
            if c == [0.0, 0.0, 0.0]:
                continue  # in paper "return None", but I think if does not make
                # sense to give up if there is no continuation of the surface at
                # radius_hit distance just in one or some of the 8 directions

            b = sqrt(dot(v_t - c, v_t - c))
            if b > radius_hit:
                continue  # in paper "return None" ...
            k_vc = 2 * b / (b ** 2 + radius_hit ** 2)
            # sign(c) = 1 if c is above the tangent plane t_v(S)
            #          -1 if c is below T_r(S)
            #           0 if c lies on T_r(S)
            sign_c = signum(dot(n_v, c - v))

            outer_product = outer(votedir, votedir)
            multiplicator = sign_c * k_vc
            B_v += multiply(outer_product, multiplicator)

        B_v /= 8
        # Decompose the symmetric matrix B_v:
        # eigenvalues are in increasing order and eigenvectors are in columns of
        # the returned quadratic matrix
        eigenvalues, eigenvectors = np.linalg.eigh(B_v)
        # Eigenvalues from highest to lowest:
        b_1 = eigenvalues[2]  # in 2D eigenvalues[1]
        b_2 = eigenvalues[1]  # in 2D eigenvalues[0]
        b_3 = eigenvalues[0]
        # Eigenvectors that correspond to the highest two eigenvalues are the
        # estimated principal directions:
        t_1 = eigenvectors[:, 2]  # in 2D eigenvectors[:, 1]
        t_2 = eigenvectors[:, 1]  # in 2D eigenvectors[:, 0]
        t_3 = eigenvectors[:, 0]
        try:
            assert(round(abs(t_3[0]), 7) == round(abs(n_v[0]), 7) and
                   round(abs(t_3[1]), 7) == round(abs(n_v[1]), 7) and
                   round(abs(t_3[2]), 7) == round(abs(n_v[2]), 7))
        except AssertionError:
            if (round(abs(t_1[0]), 7) == round(abs(n_v[0]), 7) and
                    round(abs(t_1[1]), 7) == round(abs(n_v[1]), 7) and
                    round(abs(t_1[2]), 7) == round(abs(n_v[2]), 7)):
                t_1 = t_3
                b_1 = b_3
                # t_3 = n_v; b_3 = 0
            elif (round(abs(t_2[0]), 7) == round(abs(n_v[0]), 7) and
                    round(abs(t_2[1]), 7) == round(abs(n_v[1]), 7) and
                    round(abs(t_2[2]), 7) == round(abs(n_v[2]), 7)):
                t_2 = t_3
                b_2 = b_3
                # t_3 = n_v; b_3 = 0
            else:
                print("Error: no eigenvector which equals to the normal found")
                print("t_1 = {}".format(t_1))
                print("t_2 = {}".format(t_2))
                print("t_3 = {}".format(t_3))
                print("n_v = {}".format(n_v))
                print("b_1 = {}".format(b_1))
                print("b_2 = {}".format(b_2))
                print("b_3 = {}".format(b_3))
                # add placeholders to the graph
                self.add_curvature_descriptors_to_vertex(
                    vertex_v, None, None, None, None, None, None, None, None)
        # Estimated principal curvatures:
        kappa_1 = 3 * b_1 - b_2
        kappa_2 = 3 * b_2 - b_1
        # Curvatures and directions might be interchanged:
        if kappa_1 < kappa_2:
            t_1, t_2 = t_2, t_1
            kappa_1, kappa_2 = kappa_2, kappa_1

        # Add t_1, t_2, kappa_1, kappa_2, derived Gaussian and mean curvatures
        # as well as shape index and curvedness as properties to the graph:
        gauss_curvature = calculate_gauss_curvature(kappa_1, kappa_2)
        mean_curvature = calculate_mean_curvature(kappa_1, kappa_2)
        shape_index = calculate_shape_index(kappa_1, kappa_2)
        curvedness = calculate_curvedness(kappa_1, kappa_2)
        self.add_curvature_descriptors_to_vertex(
            vertex_v, t_1, t_2, kappa_1, kappa_2, gauss_curvature,
            mean_curvature, shape_index, curvedness)

    def add_curvature_descriptors_to_vertex(
            self, vertex, t_1, t_2, kappa_1, kappa_2, gauss_curvature,
            mean_curvature, shape_index, curvedness):
        """
        Add the given curvature descriptors as vertex properties to the given
        vertex in the graph. If A property is None, a 0 value or vector is added.

        Args:
            vertex (graph_tool.Vertex): vertex where the properties should be
                added
            t_1 (ndarray): principal maximal direction vector of length 3
            t_2 (ndarray): principal minimal direction vector of length 3
            kappa_1 (float): principal maximal curvature
            kappa_2 (float): principal minimal curvature
            gauss_curvature (float): Gaussian curvature
            mean_curvature (float): mean curvature
            shape_index (float): shape index
            curvedness (float): curvedness

        Returns:
            None
        """
        self.graph.vp.t_1[vertex] = np.zeros(shape=3) if t_1 is None else t_1
        self.graph.vp.t_2[vertex] = np.zeros(shape=3) if t_2 is None else t_2
        self.graph.vp.kappa_1[vertex] = 0 if kappa_1 is None else kappa_1
        self.graph.vp.kappa_2[vertex] = 0 if kappa_2 is None else kappa_2
        self.graph.vp.gauss_curvature_VV[vertex] = (0 if gauss_curvature is None
                                                    else gauss_curvature)
        self.graph.vp.mean_curvature_VV[vertex] = (0 if mean_curvature is None
                                                   else mean_curvature)
        self.graph.vp.shape_index_VV[vertex] = (0 if shape_index is None
                                                else shape_index)
        self.graph.vp.curvedness_VV[vertex] = (0 if curvedness is None
                                               else curvedness)


class PointGraph(SurfaceGraph):
    """
    Class defining the PointGraph object, its attributes and methods.

    The constructor requires the following parameters of the underlying
    segmentation that will be used to build the graph.
    """

    def __init__(self):
        """
        Constructor of the PointGraph object.

        Returns:
            None
        """
        graphs.SegmentationGraph.__init__(self)

        # Add more "internal property maps" to the graph.
        # vertex property for storing the normal of the corresponding vertex:
        self.graph.vp.normal = self.graph.new_vertex_property("vector<float>")
        # vertex property for storing the VTK minimal curvature at the
        # corresponding triangle point:
        self.graph.vp.min_curvature = self.graph.new_vertex_property("float")
        # vertex property for storing the VTK maximal curvature at the
        # corresponding triangle point:
        self.graph.vp.max_curvature = self.graph.new_vertex_property("float")
        # vertex property for storing the VTK Gaussian curvature at the
        # corresponding triangle point:
        self.graph.vp.gauss_curvature = self.graph.new_vertex_property("float")
        # vertex property for storing the VTK mean curvature at the
        # corresponding triangle point:
        self.graph.vp.mean_curvature = self.graph.new_vertex_property("float")
        # graph property for storing the maximal triangle area:
        self.graph.gp.max_triangle_area = self.graph.new_graph_property("float")
        self.graph.gp.max_triangle_area = 0.0  # initialize
        # graph property for storing the total surface area:
        self.graph.gp.total_area = self.graph.new_graph_property("float")
        self.graph.gp.total_area = 0.0  # initialize
        # graph property for storing a list of triplet lists of vertex
        # coordinates (x, y, z) belonging to one triangle:
        self.graph.gp.triangle_points = self.graph.new_graph_property("object")
        self.graph.gp.triangle_points = []
        # graph property for storing a dictionary mapping vertex coordinates
        # (x, y, z) to a list of triangle indices sharing this point:
        self.graph.gp.point_in_triangles = self.graph.new_graph_property(
            "object")
        self.graph.gp.point_in_triangles = {}

    def build_graph_from_vtk_surface(
            self, surface, scale=(1, 1, 1), verbose=False,
            reverse_normals=False):
        """
        Builds the graph from the vtkPolyData surface, which is rescaled to
        given units according to the scale factor.

        Every vertex of the graph represents a surface triangle vertex,
        and every edge of the graph connects two adjacent vertices, just like a
        triangle edge.

        Args:
            surface (vtk.vtkPolyData): a signed surface (mesh of triangles)
                generated from the segmentation in voxels
            scale (tuple, optional): pixel size (X, Y, Z) in given units for
                scaling the surface and the graph (default (1, 1, 1))
            verbose(boolean, optional): if True (default False), some extra
                information will be printed out
            reverse_normals (boolean, optional): if True (default False), the
                triangle normals are reversed during graph generation

        Returns:
            rescaled surface to given units with VTK curvatures
            (vtk.vtkPolyData)
        """
        t_begin = time.time()

        # rescale the surface to units and update the attribute
        surface = rescale_surface(surface, scale)

        # Adding curvatures and normals to the vtkPolyData surface
        # because VTK and we (gen_surface) have the opposite normal
        # convention: VTK outwards pointing normals, we: inwards pointing
        if reverse_normals:
            invert = False
        else:
            invert = True
        surface = add_curvature_to_vtk_surface(surface, "Minimum", invert)
        surface = add_curvature_to_vtk_surface(surface, "Maximum", invert)
        surface = add_point_normals_to_vtk_surface(surface, reverse_normals)

        point_data = surface.GetPointData()
        normals = point_data.GetNormals()
        n = point_data.GetNumberOfArrays()
        min_curvatures = None
        max_curvatures = None
        gauss_curvatures = None
        mean_curvatures = None
        for i in range(n):
            if point_data.GetArrayName(i) == "Minimum_Curvature":
                min_curvatures = point_data.GetArray(i)
            elif point_data.GetArrayName(i) == "Maximum_Curvature":
                max_curvatures = point_data.GetArray(i)
            elif point_data.GetArrayName(i) == "Gauss_Curvature":
                gauss_curvatures = point_data.GetArray(i)
            elif point_data.GetArrayName(i) == "Mean_Curvature":
                mean_curvatures = point_data.GetArray(i)

        if verbose:
            # 0. Check numbers of cells and all points.
            print('{} cells'.format(surface.GetNumberOfCells()))
            print('{} points'.format(surface.GetNumberOfPoints()))

        # 1. Iterate over all cells, adding their points as vertices to the
        # graph and connecting them by edges.
        triangle_i = -1  # initialize the triangle index counter
        for i in range(surface.GetNumberOfCells()):
            if verbose:
                print('Cell number {}:'.format(i))

            # Get all points which made up the cell & check that they are 3.
            cell = surface.GetCell(i)
            points_cell = cell.GetPoints()
            if points_cell.GetNumberOfPoints() == 3:
                triangle_i += 1
                # 1a) Add each of the 3 points as vertex to the graph, if
                # it has not been added yet.
                triangle_points = []
                for j in range(0, 3):
                    x, y, z = points_cell.GetPoint(j)
                    p = (x, y, z)
                    triangle_points.append(p)
                    if p in self.graph.gp.point_in_triangles:
                        self.graph.gp.point_in_triangles[p].append(triangle_i)
                    else:
                        self.graph.gp.point_in_triangles[p] = [triangle_i]
                    if p in self.coordinates_to_vertex_index:
                        continue
                    vd = self.graph.add_vertex()  # vertex descriptor
                    self.graph.vp.xyz[vd] = [x, y, z]
                    self.coordinates_to_vertex_index[p] = \
                        self.graph.vertex_index[vd]
                    if verbose:
                        print('\tThe point ({}, {}, {}) has been added to '
                              'the graph as a vertex.'.format(x, y, z))

                    # Add VTK curvatures and normals to the graph
                    self.graph.vp.min_curvature[vd] = min_curvatures.GetTuple1(
                        cell.GetPointId(j))
                    self.graph.vp.max_curvature[vd] = max_curvatures.GetTuple1(
                        cell.GetPointId(j))
                    self.graph.vp.gauss_curvature[vd] = (
                        gauss_curvatures.GetTuple1(cell.GetPointId(j)))
                    self.graph.vp.mean_curvature[vd] = (
                        mean_curvatures.GetTuple1(cell.GetPointId(j)))
                    self.graph.vp.normal[vd] = normals.GetTuple3(
                        cell.GetPointId(j))
                self.graph.gp.triangle_points.append(triangle_points)

                # 1b) Add an edge with a distance between all 3 pairs of
                # vertices, if it has not been added yet.
                for k in range(0, 2):
                    x1, y1, z1 = points_cell.GetPoint(k)
                    p1 = (x1, y1, z1)
                    # vertex descriptor of the 1st point
                    vd1 = self.graph.vertex(
                        self.coordinates_to_vertex_index[p1])
                    for l in range(k + 1, 3):
                        x2, y2, z2 = points_cell.GetPoint(l)
                        p2 = (x2, y2, z2)
                        # vertex descriptor of the 2nd point
                        vd2 = self.graph.vertex(
                            self.coordinates_to_vertex_index[p2])
                        if not (
                              (p1, p2) in self.coordinates_pair_connected or
                              (p2, p1) in self.coordinates_pair_connected):
                            # edge descriptor
                            ed = self.graph.add_edge(vd1, vd2)
                            self.graph.ep.distance[ed] = \
                                self.distance_between_voxels(p1, p2)
                            self.coordinates_pair_connected.add((p1, p2))
                            if verbose:
                                print('\tThe neighbor points ({}, {}, {}) and '
                                      '({}, {}, {}) have been connected by an '
                                      'edge with a distance of {} pixels.'
                                      .format(x1, y1, z1, x2, y2, z2,
                                              self.graph.ep.distance[ed]))

                # 1c) Calculate area of the triangle and update the maximal area
                # and total area graph properties:
                area = triangle_area_cross_product(*np.array(triangle_points))
                if area > self.graph.gp.max_triangle_area:
                    self.graph.gp.max_triangle_area = area
                self.graph.gp.total_area += area
            else:
                print('Oops, there are {} points in cell number {}'.format(
                    points_cell.GetNumberOfPoints(), i))

        # 2. Check if the numbers of vertices and edges are as they should be:
        assert self.graph.num_vertices() == len(
            self.coordinates_to_vertex_index)
        assert self.graph.num_edges() == len(self.coordinates_pair_connected)

        t_end = time.time()
        duration = t_end - t_begin
        minutes, seconds = divmod(duration, 60)
        print('PointGraph generation took: {} min {} s'.format(
            minutes, seconds))

        return surface

    def graph_to_triangle_poly(self, verbose=False):
        """
        Generates a VTK PolyData object from the PointGraph object with
        triangle-cells representing the surface triangles.

        Args:
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out

        Returns:
            vtk.vtkPolyData with triangle-cells
        """
        if self.graph.num_vertices() > 0:
            # Initialization
            poly_triangles = vtk.vtkPolyData()
            points = vtk.vtkPoints()
            vertex_arrays = list()  # should become point's properties
            # Vertex property arrays
            for prop_key in list(self.graph.vp.keys()):
                data_type = self.graph.vp[prop_key].value_type()
                if (data_type != 'string' and data_type != 'python::object' and
                        prop_key != 'points'):  # and prop_key != 'xyz'
                    if verbose:
                        print('\nvertex property key: {}'.format(prop_key))
                        print('value type: {}'.format(data_type))
                    if data_type[0:6] != 'vector':  # scalar
                        num_components = 1
                    else:  # vector
                        num_components = len(
                            self.graph.vp[prop_key][self.graph.vertex(0)])
                    array = TypesConverter().gt_to_vtk(data_type)
                    array.SetName(prop_key)
                    if verbose:
                        print('number of components: {}'.format(num_components))
                    array.SetNumberOfComponents(num_components)
                    vertex_arrays.append(array)
            if verbose:
                print('\nvertex arrays length: {}'.format(len(vertex_arrays)))

            # Geometry
            # dictionary of points with a key (x, y, z) and the index in VTK
            # points list as a value
            points_dict = {}
            for vd in self.graph.vertices():  # they are the points!
                [x, y, z] = self.graph.vp.xyz[vd]
                # add the new point everywhere & update the index
                i = self.graph.vertex_index[vd]
                points.InsertPoint(i, x, y, z)
                points_dict[(x, y, z)] = i

                for array in vertex_arrays:
                    prop_key = array.GetName()
                    n_comp = array.GetNumberOfComponents()
                    data_type = self.graph.vp[prop_key].value_type()
                    data_type = TypesConverter().gt_to_numpy(data_type)
                    array.InsertNextTuple(self.get_vertex_prop_entry(
                        prop_key, vd, n_comp, data_type))
            if verbose:
                print('number of points: {}'.format(points.GetNumberOfPoints()))

            # Topology
            # Triangles
            triangles = vtk.vtkCellArray()
            for triangle_points in self.graph.gp.triangle_points:
                # triangle_points is a list of three (x, y, z)
                triangle = vtk.vtkTriangle()
                # The first parameter is the index of the triangle vertex which
                # is ALWAYS 0-2.
                # The second parameter is the index into the point (geometry)
                # array, so this can range from 0-(NumPoints-1)
                triangle.GetPointIds().SetId(
                    0, points_dict[triangle_points[0]])
                triangle.GetPointIds().SetId(
                    1, points_dict[triangle_points[1]])
                triangle.GetPointIds().SetId(
                    2, points_dict[triangle_points[2]])
                triangles.InsertNextCell(triangle)
            if verbose:
                print('number of triangle cells: {}'.format(
                    triangles.GetNumberOfCells()))

            # vtkPolyData construction
            poly_triangles.SetPoints(points)
            poly_triangles.SetPolys(triangles)
            for array in vertex_arrays:
                poly_triangles.GetPointData().AddArray(array)

            return poly_triangles

        else:
            print("The graph is empty!")
            return None

    # * The following PointGraph method is implementing with adaptations
    # the first part of the first step of normal vector voting algorithm of
    # Page et al., 2002. *
    def collect_normal_votes(self, vertex_v_ind, g_max, a_max, sigma):
        """
        For a vertex v, collects the normal votes of all triangles within its
        geodesic neighborhood and calculates the weighted covariance matrix sum
        V_v.

        Implements equations (6), illustrated in figure 6(b), (7) and (8) from
        the paper of Page et al., 2002.

        More precisely, a normal vote n_i of each triangle centroid i (whose
        three points are lying within the geodesic neighborhood of vertex v) is
        calculated using the normal n_c_i assigned to the triangle i. Then, each
        vote is represented by a covariance matrix V_i and votes are collected
        as a weighted matrix sum V_v, where each vote is weighted depending on
        the area of triangle i and the geodesic distance of its centroid c_i
        from v.

        Here, c_i is centroid of a triangle (vertex of TriangleGraph) and v is a
        triangle vertex (vertex of PointGraph), like in Page's approach).

        Args:
            vertex_v_ind (int): index of the vertex v in the surface
                triangle-graph for which the votes are collected
            g_max (float): the maximal geodesic distance in units of the graph
            a_max (float): the area of the largest triangle in the surface
                triangle-graph
            sigma (float): sigma, defined as 3*sigma = g_max, so that votes
                beyond the neighborhood can be ignored

        Returns:
            - number of geodesic neighbors of vertex v
            - the 3x3 symmetric matrix V_v (numpy.ndarray)
        """
        # To spare function referencing every time in the following for loop:
        vertex = self.graph.vertex
        normal = self.graph.vp.normal
        array = np.array
        xyz = self.graph.vp.xyz
        sqrt = math.sqrt
        dot = np.dot
        outer = np.multiply.outer
        exp = math.exp
        point_in_triangles = self.graph.gp.point_in_triangles
        calculate_geodesic_distance = self._calculate_geodesic_distance
        # acos = nice_acos

        # Get the coordinates of vertex v (as numpy array):
        vertex_v = vertex(vertex_v_ind)
        v = array(xyz[vertex_v])
        # n_v = array(normal[vertex_v])  # original normal at v

        # Find the neighboring vertices of vertex v to be returned:
        if vertex_v_ind == 0:
            print("Calling find_geodesic_neighbors_exact")
        neighbor_idx_to_dist = self.find_geodesic_neighbors_exact(
            vertex_v, g_max, verbose=False, debug=False)
        try:
            assert len(neighbor_idx_to_dist) > 0
        except AssertionError:
            print("\nWarning: the vertex v = {} has 0 neighbors. It will be "
                  "ignored later.".format(v))
            # return a placeholder instead of V_v
            return 0, np.zeros(shape=(3, 3))

        # Initialize the weighted matrix sum of all votes for vertex v to be
        # calculated and returned:
        V_v = np.zeros(shape=(3, 3))

        # Find the neighboring triangles of vertex v:
        neighboring_triangles_of_v = {}  # triangle_idx -> vertex_ids
        for idx_v_i in list(neighbor_idx_to_dist.keys()):
            # Get neighboring vertex v_i and its coordinates (as numpy array):
            vertex_v_i = vertex(idx_v_i)
            v_i = array(xyz[vertex_v_i])
            triangle_ids_of_v_i = point_in_triangles[tuple(v_i)]
            for triangle_idx_of_v_i in triangle_ids_of_v_i:
                if triangle_idx_of_v_i in neighboring_triangles_of_v:
                    neighboring_triangles_of_v[triangle_idx_of_v_i].append(
                        idx_v_i)
                else:
                    neighboring_triangles_of_v[triangle_idx_of_v_i] = [idx_v_i]
        # Exclude neighboring triangles of vertex v with less than 3 vertices
        for triangle_idx, vertex_ids in list(neighboring_triangles_of_v.items()):
            if len(vertex_ids) < 3:
                del neighboring_triangles_of_v[triangle_idx]

        # Let each of the neighboring triangles c_i to cast a vote on vertex v:
        # except for those with almost opposite normals
        for idx_c_i, ids_v_i in list(neighboring_triangles_of_v.items()):
            # Calculate the normal of triangle c_i, n_c_i:
            points_xyz = [array(xyz[vertex(idx_v_i)]) for idx_v_i in ids_v_i]
            ref_normal = array(normal[vertex(ids_v_i[0])])
            n_c_i = triangle_normal(ref_normal, *points_xyz)

            # Calculate center of the triangle c_i and normalized vc_i vector:
            c_i = triangle_center(*points_xyz)
            vc_i = c_i - v
            vc_i_len = sqrt(dot(vc_i, vc_i))
            vc_i_norm = vc_i / vc_i_len

            # theta_i is the angle between the vectors n_c_i and vc_i
            cos_theta_i = - (dot(n_c_i, vc_i)) / vc_i_len

            # Calculate the normal vote n_i of c_i on v:
            n_i = n_c_i + 2 * cos_theta_i * vc_i_norm

            # Covariance matrix containing one vote of c_i on v:
            V_i = outer(n_i, n_i)

            # Calculate the weight depending on the area of the neighboring
            # triangle i, a_i, and the geodesic distance to its center c_i from
            # vertex v, g_c_i:
            a_i = triangle_area_cross_product(*points_xyz)
            # Geodesic distances to the three vertices of the triangle i:
            g_v_i_s = [neighbor_idx_to_dist[idx_v_i] for idx_v_i in ids_v_i]
            # Find two triangle vertices among them that are closer to vertex v:
            # Add all 3 vertices and remove the first one with maximal distance
            # (because two vertices can have equal distances from origin)
            v_i_s = [vertex(idx_v_i) for idx_v_i in ids_v_i]
            for idx_v_i in ids_v_i:
                if neighbor_idx_to_dist[idx_v_i] == max(g_v_i_s):
                    v_i_s.remove(vertex(idx_v_i))
                    break
            [v_a, v_b] = v_i_s
            # Calculate g_c_i using these two vertices, a and b, forming an
            # imaginary triangle with vertices a, b and c_i:
            g_c_i = calculate_geodesic_distance(
                v_a, v_b, c_i, neighbor_idx_to_dist, verbose=False)
            w_i = a_i / a_max * exp(- g_c_i / sigma)

            # Weigh V_i and add it to the weighted matrix sum:
            V_v += w_i * V_i

        return len(neighbor_idx_to_dist), V_v


class TriangleGraph(SurfaceGraph):
    """
    Class defining the TriangleGraph object, its attributes and methods.

    The constructor requires the following parameters of the underlying
    segmentation that will be used to build the graph.
    """

    def __init__(self):
        """
        Constructor of the TriangleGraph object.

        Returns:
            None
        """
        graphs.SegmentationGraph.__init__(self)

        # Add more "internal property maps" to the graph.
        # vertex property for storing the area of the corresponding triangle:
        self.graph.vp.area = self.graph.new_vertex_property("float")
        # vertex property for storing the normal of the corresponding triangle:
        self.graph.vp.normal = self.graph.new_vertex_property("vector<float>")
        # vertex property for storing the VTK minimal curvature at the
        # corresponding triangle:
        self.graph.vp.min_curvature = self.graph.new_vertex_property("float")
        # vertex property for storing the VTK maximal curvature at the
        # corresponding triangle:
        self.graph.vp.max_curvature = self.graph.new_vertex_property("float")
        # vertex property for storing the VTK Gaussian curvature at the
        # corresponding triangle:
        self.graph.vp.gauss_curvature = self.graph.new_vertex_property("float")
        # vertex property for storing the VTK mean curvature at the
        # corresponding triangle:
        self.graph.vp.mean_curvature = self.graph.new_vertex_property("float")
        # vertex property for storing the coordinates of the 3 points of the
        # corresponding triangle:
        self.graph.vp.points = self.graph.new_vertex_property("object")
        # edge property storing the "strength" property of the edge: 1 for a
        # "strong" or 0 for a "weak" one:
        self.graph.ep.is_strong = self.graph.new_edge_property("int")
        # graph property for storing the maximal triangle area:
        self.graph.gp.max_triangle_area = self.graph.new_graph_property("float")
        self.graph.gp.max_triangle_area = 0.0  # initialize
        # graph property for storing the total surface area:
        self.graph.gp.total_area = self.graph.new_graph_property("float")

        self.point_in_cells = {}
        """dict: a dictionary mapping a point coordinates (x, y, z) to a list of
        triangle-cell indices sharing this point.
        """

        self.triangle_cell_ids = {}
        """dict: a dictionary of all added triangle cell indices, mapping them
        to their graph vertex indices"""

    def build_graph_from_vtk_surface(
            self, surface, scale=(1, 1, 1), vtk_curv=True, verbose=False,
            reverse_normals=False):
        """
        Builds the graph from the vtkPolyData surface, which is rescaled to
        given units according to the scale factor.

        Every vertex of the graph represents the center of a surface triangle,
        and every edge of the graph connects two adjacent triangles. There are
        two types of edges: a "strong" edge if the adjacent triangles share two
        triangle edges and a "weak" edge if they share only one edge.

        Args:
            surface (vtk.vtkPolyData): a signed surface (mesh of triangles)
                generated from the segmentation in voxels
            scale (tuple, optional): pixel size (X, Y, Z) in given units for
                scaling the surface and the graph (default (1, 1, 1))
            vtk_curv (boolean, optional): if True (default), add VTK curvatures
                at each triangle center by averaging the values at the three
                triangle vertices. Setting this option to False can save time
                for big surfaces, if VTK comparison is not needed.
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out
            reverse_normals (boolean, optional): if True (default False), the
                triangle normals are reversed during graph generation

        Returns:
            rescaled surface to given units with VTK curvatures
            (vtk.vtkPolyData)
        """
        t_begin = time.time()

        # 1. Preparation
        if scale != (1, 1, 1):
            # rescale the surface to units
            surface = rescale_surface(surface, scale)

        if verbose:
            # Check numbers of cells and all points.
            print('{} cells'.format(surface.GetNumberOfCells()))
            print('{} points'.format(surface.GetNumberOfPoints()))

        min_curvatures = None
        max_curvatures = None
        gauss_curvatures = None
        mean_curvatures = None
        if vtk_curv:
            # Adding curvatures to the vtkPolyData surface
            # because VTK and we (gen_surface) have the opposite normal
            # convention: VTK outwards pointing normals, we: inwards pointing
            if reverse_normals:
                invert = False
            else:
                invert = True
            surface = add_curvature_to_vtk_surface(surface, "Minimum", invert)
            surface = add_curvature_to_vtk_surface(surface, "Maximum", invert)
            point_data = surface.GetPointData()
            n = point_data.GetNumberOfArrays()
            for i in range(n):
                if point_data.GetArrayName(i) == "Minimum_Curvature":
                    min_curvatures = point_data.GetArray(i)
                elif point_data.GetArrayName(i) == "Maximum_Curvature":
                    max_curvatures = point_data.GetArray(i)
                elif point_data.GetArrayName(i) == "Gauss_Curvature":
                    gauss_curvatures = point_data.GetArray(i)
                elif point_data.GetArrayName(i) == "Mean_Curvature":
                    mean_curvatures = point_data.GetArray(i)

        # 2. Add each triangle cell as a vertex to the graph. Ignore the
        # non-triangle cells and cell with area equal to zero.
        for cell_id in range(surface.GetNumberOfCells()):
            # Get the cell i and check if it's a triangle:
            cell = surface.GetCell(cell_id)
            if not isinstance(cell, vtk.vtkTriangle):
                print('Oops, the cell number {} is not a vtkTriangle but a {}! '
                      'It will be ignored.'.format(
                       cell_id, cell.__class__.__name__))
                continue
            if verbose:
                print('Triangle cell number {}'.format(cell_id))

            # Initialize a list for storing the points coordinates making
            # out the cell
            points_xyz = []

            # Get the 3 points which made up the triangular cell:
            points_cell = cell.GetPoints()

            # Calculate the area of the triangle i and update the maximal area
            # graph property:
            area = cell.TriangleArea(points_cell.GetPoint(0),
                                     points_cell.GetPoint(1),
                                     points_cell.GetPoint(2))
            if area > self.graph.gp.max_triangle_area:
                self.graph.gp.max_triangle_area = area
            try:
                assert(area > 0)
            except AssertionError:
                print('\tThe cell {} cannot be added to the graph as a vertex, '
                      'because the triangle area is not positive, but is {}.'
                      .format(cell_id, area))
                continue

            # Calculate the centroid of the triangle:
            for j in range(0, 3):
                x, y, z = points_cell.GetPoint(j)
                # Add each point j as a key in point_in_cells and cell index
                # to the value list:
                point_j = (x, y, z)
                if point_j in self.point_in_cells:
                    self.point_in_cells[point_j].append(cell_id)
                else:
                    self.point_in_cells[point_j] = [cell_id]

                # Add each point j into the points coordinates list
                points_xyz.append(np.array([x, y, z]))
            center_xyz = triangle_center(*points_xyz)

            # Calculate the normal of the triangle i;
            normal = np.zeros(shape=3)
            cell.ComputeNormal(points_cell.GetPoint(0),
                               points_cell.GetPoint(1),
                               points_cell.GetPoint(2), normal)
            if reverse_normals:
                normal *= -1

            # Add the centroid as vertex to the graph, setting its properties:
            vd = self.graph.add_vertex()  # vertex descriptor
            vd_id = self.graph.vertex_index[vd]
            self.graph.vp.xyz[vd] = center_xyz
            self.coordinates_to_vertex_index[tuple(center_xyz)] = vd_id
            self.graph.vp.area[vd] = area
            self.graph.vp.normal[vd] = normal
            self.graph.vp.points[vd] = points_xyz

            if vtk_curv:
                # Get the min, max, Gaussian and mean curvatures (calculated by
                # VTK) for each of 3 points of the triangle i and calculate the
                # average curvatures:
                avg_min_curvature = np.average(
                    [min_curvatures.GetTuple1(
                        cell.GetPointId(j)) for j in range(0, 3)])
                avg_max_curvature = np.average(
                    [max_curvatures.GetTuple1(
                        cell.GetPointId(j)) for j in range(0, 3)])
                avg_gauss_curvature = np.average(
                    [gauss_curvatures.GetTuple1(
                        cell.GetPointId(j)) for j in range(0, 3)])
                avg_mean_curvature = np.average(
                    [mean_curvatures.GetTuple1(
                        cell.GetPointId(j)) for j in range(0, 3)])
                # Add the properties to the graph vertex:
                self.graph.vp.min_curvature[vd] = avg_min_curvature
                self.graph.vp.max_curvature[vd] = avg_max_curvature
                self.graph.vp.gauss_curvature[vd] = avg_gauss_curvature
                self.graph.vp.mean_curvature[vd] = avg_mean_curvature

            if verbose:
                print('\tThe triangle centroid {} has been added to the graph '
                      'as a vertex. Triangle area = {}, normal = {},\n'
                      'points = {}.'.format(
                       self.graph.vp.xyz[vd], self.graph.vp.area[vd],
                       self.graph.vp.normal[vd], self.graph.vp.points[vd]))

            self.triangle_cell_ids[cell_id] = vd_id

        # 3. Add edges for each cell / vertex.
        for cell_id, i in self.triangle_cell_ids.items():
            # Note: i corresponds to the vertex number of each cell
            cell = surface.GetCell(cell_id)
            if verbose:
                print('(Triangle) cell number {}:'.format(cell_id))

            # Find the "neighbor cells" and how many points they share with
            # cell i (1 or 2) as follows.
            # For each point j of cell i, iterate over the neighbor cells
            # sharing that point (excluding the cell i).
            # Add each neighbor cell to the neighbor_cells dict if it is not
            # there yet and add 1 into the shared_points list.
            # Otherwise, find the index of the cell in neighbor_cells and
            # increase the counter of shared_points at the same index.
            points_cell = cell.GetPoints()
            neighbor_cells = {}
            shared_points = []
            idx_new = 0  # counter of newly added neighbor cells of point j
            for j in range(points_cell.GetNumberOfPoints()):
                point_j = points_cell.GetPoint(j)
                for neighbor_cell_id in self.point_in_cells[point_j]:
                    if neighbor_cell_id != cell_id:
                        if neighbor_cell_id not in neighbor_cells:
                            neighbor_cells[neighbor_cell_id] = idx_new
                            shared_points.append(1)
                            idx_new += 1
                        else:
                            idx = neighbor_cells[neighbor_cell_id]
                            shared_points[idx] += 1
            if verbose:
                print("has {} neighbor cells".format(len(neighbor_cells)))

            # Get the vertex descriptor representing the cell i (vertex i):
            vd_i = self.graph.vertex(i)

            # Get the coordinates of the vertex i:
            p_i = self.graph.vp.xyz[vd_i]  # a list
            p_i = tuple(p_i)

            # Iterate over the ready neighbor_cells dict and shared_points list,
            # connecting cell i with a neighbor cell x with a "strong" edge
            # if they share 2 edges and with a "weak" edge otherwise (if
            # they share only 1 edge).
            for neighbor_cell_id, idx in neighbor_cells.items():
                # Get the vertex descriptor representing the cell x:
                # vertex index of the current neighbor cell
                x = self.triangle_cell_ids[neighbor_cell_id]
                # vertex descriptor of the current neighbor cell, vertex x
                vd_x = self.graph.vertex(x)

                # Get the coordinates of the vertex x:
                p_x = self.graph.vp.xyz[vd_x]
                p_x = tuple(p_x)

                # Add an edge between the vertices i and x, if it has not
                # been added yet:
                if not (((p_i, p_x) in self.coordinates_pair_connected) or
                        ((p_x, p_i) in self.coordinates_pair_connected)):
                    ed = self.graph.add_edge(vd_i, vd_x)  # edge descriptor
                    self.coordinates_pair_connected.add((p_i, p_x))

                    # Add the distance of the edge
                    self.graph.ep.distance[ed] = self.distance_between_voxels(
                        p_i, p_x)

                    # Assign the "strength" property to the edge as
                    # explained above:
                    if shared_points[idx] == 2:
                        is_strong = 1
                        strength = 'strong'
                    else:
                        is_strong = 0
                        strength = 'weak'
                    self.graph.ep.is_strong[ed] = is_strong
                    if verbose:
                        print('\tThe neighbor vertices ({}, {}, {}) and '
                              '({}, {}, {}) have been connected by a {} '
                              'edge with a distance of {} pixels.'.format(
                               p_i[0], p_i[1], p_i[2],
                               p_x[0], p_x[1], p_x[2], strength,
                               self.graph.ep.distance[ed]))

        # 4. Check if the numbers of vertices and edges are as they should be:
        assert self.graph.num_vertices() == len(self.triangle_cell_ids)
        assert self.graph.num_edges() == len(self.coordinates_pair_connected)
        if verbose:
            print('Real number of unique points: {}'.format(
                len(self.point_in_cells)))

        # 5. Calculate the total surface area and store as graph property:
        triangle_areas = self.graph.vp.area.get_array()
        self.graph.gp.total_area = np.sum(triangle_areas)

        t_end = time.time()
        duration = t_end - t_begin
        minutes, seconds = divmod(duration, 60)
        print('TriangleGraph generation took: {} min {} s'.format(
            minutes, seconds))

        return surface

    def graph_to_triangle_poly(self, verbose=False):
        """
        Generates a VTK PolyData object from the TriangleGraph object with
        triangle-cells representing the surface triangles.

        Args:
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out

        Returns:
            vtk.vtkPolyData with triangle-cells
        """
        if self.graph.num_vertices() > 0:
            # Initialization
            poly_triangles = vtk.vtkPolyData()
            points = vtk.vtkPoints()
            vertex_arrays = list()
            # Vertex property arrays
            for prop_key in list(self.graph.vp.keys()):
                data_type = self.graph.vp[prop_key].value_type()
                if (data_type != 'string' and data_type != 'python::object' and
                        prop_key != 'points'):  # and prop_key != 'xyz'
                    if verbose:
                        print('\nvertex property key: {}'.format(prop_key))
                        print('value type: {}'.format(data_type))
                    if data_type[0:6] != 'vector':  # scalar
                        num_components = 1
                    else:  # vector
                        num_components = len(
                            self.graph.vp[prop_key][self.graph.vertex(0)])
                    array = TypesConverter().gt_to_vtk(data_type)
                    array.SetName(prop_key)
                    if verbose:
                        print('number of components: {}'.format(num_components))
                    array.SetNumberOfComponents(num_components)
                    vertex_arrays.append(array)
            if verbose:
                print('\nvertex arrays length: {}'.format(len(vertex_arrays)))

            # Geometry
            # lut[vertex_index, triangle_point_index*] = point_array_index**
            # *ALWAYS 0-2, **0-(NumPoints-1)
            lut = np.zeros(shape=(self.graph.num_vertices(), 3), dtype=int)
            i = 0  # next new point index
            # dictionary of points with a key (x, y, z) and the index in VTK
            # points list as a value
            points_dict = {}
            for vd in self.graph.vertices():
                # enumerate over the 3 points of the triangle (vertex)
                for j, [x, y, z] in enumerate(self.graph.vp.points[vd]):
                    # add the new point everywhere & update the index
                    if (x, y, z) not in points_dict:
                        points.InsertPoint(i, x, y, z)
                        lut[self.graph.vertex_index[vd], j] = i
                        points_dict[(x, y, z)] = i
                        i += 1
                    else:  # reference the old point index only in the lut
                        lut[self.graph.vertex_index[vd], j] = points_dict[
                            (x, y, z)]
            if verbose:
                print('number of points: {}'.format(points.GetNumberOfPoints()))

            # Topology
            # Triangles
            triangles = vtk.vtkCellArray()
            for vd in self.graph.vertices():  # vd = vertex descriptor
                # storing triangles of type Triangle:
                triangle = vtk.vtkTriangle()
                # The first parameter is the index of the triangle vertex which
                # is ALWAYS 0-2.
                # The second parameter is the index into the point (geometry)
                # array, so this can range from 0-(NumPoints-1)
                triangle.GetPointIds().SetId(
                    0, lut[self.graph.vertex_index[vd], 0])
                triangle.GetPointIds().SetId(
                    1, lut[self.graph.vertex_index[vd], 1])
                triangle.GetPointIds().SetId(
                    2, lut[self.graph.vertex_index[vd], 2])
                triangles.InsertNextCell(triangle)
                for array in vertex_arrays:
                    prop_key = array.GetName()
                    n_comp = array.GetNumberOfComponents()
                    data_type = self.graph.vp[prop_key].value_type()
                    data_type = TypesConverter().gt_to_numpy(data_type)
                    array.InsertNextTuple(self.get_vertex_prop_entry(
                        prop_key, vd, n_comp, data_type))
            if verbose:
                print('number of triangle cells: {}'.format(
                    triangles.GetNumberOfCells()))

            # vtkPolyData construction
            poly_triangles.SetPoints(points)
            poly_triangles.SetPolys(triangles)
            for array in vertex_arrays:
                poly_triangles.GetCellData().AddArray(array)

            return poly_triangles

        else:
            print("The graph is empty!")
            return None

    def find_graph_border(self, purge=False):
        """
        Finds vertices at the graph border, defined as such having less than 3
        strong edges.

        Args:
            purge (boolean, optional): if True, those vertices and their edges
                will be filtered out permanently, if False (default) no
                filtering will be done

        Returns:
            list of indices of vertices at the graph border
        """
        if "is_on_border" not in self.graph.vertex_properties:
            print('Finding vertices at the graph border...')
            # Add a vertex property for storing the number of strong edges:
            self.graph.vp.num_strong_edges = self.graph.new_vertex_property(
                "int")
            # Sum up the "strong" edges coming out of each vertex and add them
            # to the new property:
            incident_edges_op(self.graph, "out", "sum", self.graph.ep.is_strong,
                              self.graph.vp.num_strong_edges)

            # Add a boolean vertex property telling whether a vertex is on
            # border:
            self.graph.vp.is_on_border = self.graph.new_vertex_property(
                "boolean")
            # indices of vertices with less than 3 strong edges (= vertices on
            # border)
            border_vertices_indices = np.where(
                self.graph.vp.num_strong_edges.a < 3)[0]
            self.graph.vp.is_on_border.a = np.zeros(
                shape=self.graph.num_vertices())
            self.graph.vp.is_on_border.a[border_vertices_indices] = 1
        else:
            border_vertices_indices = np.where(
                self.graph.vp.is_on_border.a == 1)[0]
        print('{} vertices are at the graph border.'.format(
            len(border_vertices_indices)))

        if purge is True:
            print('Filtering out the vertices at the graph borders and their '
                  'edges...')
            # Set the filter to get only vertices NOT on border.
            self.graph.set_vertex_filter(self.graph.vp.is_on_border.t(lambda x: 1-x))
            # Purge the filtered out vertices and edges permanently from the
            # graph:
            self.graph.purge_vertices()
            # Remove the properties used for the filtering that are no longer
            # true:
            del self.graph.vertex_properties["num_strong_edges"]
            del self.graph.vertex_properties["is_on_border"]
            # Update graph's dictionary coordinates_to_vertex_index:
            self.update_coordinates_to_vertex_index()

        return border_vertices_indices

    def find_vertices_near_border(self, b, purge=False):
        """
        Finds vertices that are within a given distance to the graph border.

        Args:
            b (float): distance from border in given units
            purge (boolean, optional): if True, those vertices and their edges
                will be filtered out permanently; if False (default), no
                filtering will be done

        Returns:
            None
        """
        if "is_near_border" not in self.graph.vertex_properties:
            border_vertices_indices = self.find_graph_border()

            print('For each graph border vertex, finding vertices within '
                  'geodesic distance {} to it...'.format(b))
            vertex_id_within_b_to_border = dict()
            for border_v_i in border_vertices_indices:
                border_v = self.graph.vertex(border_v_i)
                dist_border_v = shortest_distance(
                    self.graph, source=border_v, target=None,
                    weights=self.graph.ep.distance, max_dist=b)
                dist_border_v = dist_border_v.get_array()

                idxs = np.where(dist_border_v <= b)[0]
                for idx in idxs:
                    dist = dist_border_v[idx]
                    try:
                        vertex_id_within_b_to_border[idx] = min(
                            dist, vertex_id_within_b_to_border[idx])
                    except KeyError:
                        vertex_id_within_b_to_border[idx] = dist
            print('{} vertices are within distance {} to the graph border.'
                  .format(len(vertex_id_within_b_to_border), b))

            # Add a boolean vertex property telling whether a vertex is within
            # distance b to border:
            self.graph.vp.is_near_border = self.graph.new_vertex_property(
                "boolean")
            for v in self.graph.vertices():
                v_i = self.graph.vertex_index[v]
                if v_i in vertex_id_within_b_to_border:
                    self.graph.vp.is_near_border[v] = 1
                else:
                    self.graph.vp.is_near_border[v] = 0

        if purge is True:
            print('Filtering out those vertices and their edges...')
            # Set the filter to get only vertices NOT within distance b to
            # border.
            self.graph.set_vertex_filter(self.graph.vp.is_near_border.t(lambda x: 1-x))
            # Purge filtered out vertices and edges from the graph:
            self.graph.purge_vertices()
            # Update graph's dictionary coordinates_to_vertex_index:
            self.update_coordinates_to_vertex_index()

    def find_vertices_outside_mask(
            self, mask, scale, label=1, allowed_dist=0):
        """
        Finds vertices that are outside a mask.

        This means that their scaled back to pixels coordinates are further away
        than an allowed distance in pixels to a mask voxel with the given label.

        Args:
            mask (numpy.ndarray): 3D mask of the segmentation from which the
                underlying surface was created
            scale (tuple): pixel size (X, Y, Z) in given units
            label (int, optional): the label in the mask to be considered
                (default 1)
            allowed_dist (int, optional): allowed distance in pixels between a
                voxel coordinate and a mask voxel (default 0)

        Returns:
            None
        """
        if isinstance(mask, np.ndarray):
            print('\nFinding vertices outside the membrane mask...')
            # Add a boolean vertex property telling whether a vertex is outside
            # the mask:
            self.graph.vp.is_outside_mask = self.graph.new_vertex_property(
                "boolean")
            # Invert the boolean matrix, because distance_transform_edt
            # calculates distances from '0's, not from '1's!
            maskd = ndimage.morphology.distance_transform_edt(
                np.invert(mask == label))

            num_vertices_outside_mask = 0
            for v in self.graph.vertices():
                v_xyz = self.graph.vp.xyz[v]
                v_pixel_x = int(round(v_xyz[0] / scale[0]))
                v_pixel_y = int(round(v_xyz[1] / scale[1]))
                v_pixel_z = int(round(v_xyz[2] / scale[2]))
                try:
                    if maskd[v_pixel_x, v_pixel_y, v_pixel_z] > allowed_dist:
                        self.graph.vp.is_outside_mask[v] = 1
                        num_vertices_outside_mask += 1
                    else:
                        self.graph.vp.is_outside_mask[v] = 0
                except IndexError:
                    print("IndexError happened. Vertex with coordinates "
                          "({}, {}, {})".format(v_xyz[0], v_xyz[1], v_xyz[2]))
                    print("was transformed to pixel ({}, {}, {}),".format(
                        v_pixel_x, v_pixel_y, v_pixel_z))
                    print("which is not inside the mask with shape ({}, {}, {})"
                          .format(maskd.shape[0], maskd.shape[1],
                                  maskd.shape[2]))
            print('{} vertices are further away than {} pixel to the mask.'
                  .format(num_vertices_outside_mask, allowed_dist))

        else:
            raise pexceptions.PySegInputError(
                expr='find_vertices_outside_mask (TriangleGraph)',
                msg="A a 3D numpy ndarray object required as the first input.")

    def find_vertices_near_border_and_outside_mask(
            self, b, mask, scale, label=1, allowed_dist=0,
            purge=False):
        """
        Finds vertices that are within distance to the graph border and outside
        a mask.

        Outside mask means that scaled back to pixels vertices coordinates are
        further than the allowed distance in pixels to a mask voxel with the
        given label.

        Args:
            b (float): distance from border in units of the graph
            mask (numpy.ndarray): 3D mask of the segmentation from which the
                underlying surface was created
            scale (tuple): pixel size (X, Y, Z) in given units
            label (int, optional): the label in the mask to be considered
                (default 1)
            allowed_dist (int, optional): allowed distance in pixels between a
                voxel coordinate and a mask voxel (default 0)
            purge (boolean, optional): if True, those vertices and their edges
                will be filtered out permanently; if False (default), no
                filtering will be done

        Returns:
            None
        """
        # don't remove vertices near border, because have to intersect with the
        # mask first!
        self.find_vertices_near_border(b, purge=False)
        self.find_vertices_outside_mask(mask, scale, label=label,
                                        allowed_dist=allowed_dist)

        # Add a boolean vertex property telling whether a vertex within distance
        # b to border and outside mask:
        self.graph.vp.is_near_border_and_outside_mask = \
            self.graph.new_vertex_property("boolean")
        num_vertices_near_border_and_outside_mask = 0
        for v in self.graph.vertices():
            if (self.graph.vp.is_near_border[v] == 1 and
                    self.graph.vp.is_outside_mask[v] == 1):
                self.graph.vp.is_near_border_and_outside_mask[v] = 1
                num_vertices_near_border_and_outside_mask += 1
            else:
                self.graph.vp.is_near_border_and_outside_mask[v] = 0
        print('{} vertices are within distance {} to the graph border and '
              'further than {} pixel from the mask.'.format(
               num_vertices_near_border_and_outside_mask, b, allowed_dist))

        if purge is True:
            print('Filtering out those vertices and their edges...')
            # Set the filter to get only vertices NOT {near border and outside
            # mask}.
            self.graph.set_vertex_filter(
                self.graph.vp.is_near_border_and_outside_mask.t(lambda x: 1-x))
            # Purge filtered out vertices and edges permanently from the graph:
            self.graph.purge_vertices()
            # Remove the properties used for filtering that are no longer true:
            del self.graph.vertex_properties["num_strong_edges"]
            del self.graph.vertex_properties["is_on_border"]
            del self.graph.vertex_properties["is_near_border"]
            # until here as in find_vertices_near_border, because not purged
            del self.graph.vertex_properties["is_outside_mask"]
            del self.graph.vertex_properties["is_near_border_and_outside_mask"]
            # Update graph's dictionary coordinates_to_vertex_index:
            self.update_coordinates_to_vertex_index()

    def find_largest_connected_component(self, replace=False):
        """
        Finds the largest connected component (lcc) of the graph.

        Args:
            replace (boolean, optional): if True (default False) and the lcc has
                less vertices, the graph is replaced by its lcc

        Returns:
            the lcc of the graph (graph_tool.GraphView)

        Note:
            Use lcc only if you are sure that the segmentation region was not
            split into multiple parts during surface generation of subsequent
            surface cleaning.
        """
        print('Total number of vertices in the graph: {}'.format(
            self.graph.num_vertices()))
        is_in_lcc = label_largest_component(self.graph)
        lcc = GraphView(self.graph, vfilt=is_in_lcc)
        print('Number of vertices in the largest connected component of the '
              'graph: {}'.format(lcc.num_vertices()))

        if replace is True and lcc.num_vertices() < self.graph.num_vertices():
            print('Filtering out those vertices and edges not belonging to the '
                  'largest connected component...')
            # Set the filter to get only vertices belonging to the lcc.
            self.graph.set_vertex_filter(is_in_lcc)
            # Purge filtered out vertices and edges permanently from the graph:
            self.graph.purge_vertices()
            # Update graph's dictionary coordinates_to_vertex_index:
            self.update_coordinates_to_vertex_index()

        return lcc

    def find_small_connected_components(self, threshold=100, purge=False,
                                        verbose=False):
        """
        Finds small connected components of the graph that are below a given
        threshold size in voxels.

        Args:
            threshold (int, optional): threshold size in voxels (default 100),
                below which connected components are considered small
            purge (boolean, optional): if True (default False) and small
                components were found, their vertices and edges will be filtered
                out permanently; otherwise no filtering will be done
            verbose (boolean, optional): if True (default False), some extra
                information will be printed out

        Returns:
            None
        """
        t_begin = time.time()

        comp_labels_map, sizes = label_components(self.graph)
        small_components = [i for i, size in enumerate(sizes)
                            if size < threshold]
        print("The graph has {} components, {} of them have size < {}".format(
            len(sizes), len(small_components), threshold))
        if verbose:
            print("Sizes of components:")
            print(sizes)

        if len(small_components) > 0:
            # Add a boolean vertex property telling whether a vertex belongs to
            # a small component with size below the threshold:
            self.graph.vp.small_component = \
                self.graph.new_vertex_property("boolean")
            num_vertices_in_small_components = 0
            for v in self.graph.vertices():
                if comp_labels_map[v] in small_components:
                    self.graph.vp.small_component[v] = 1
                    num_vertices_in_small_components += 1
                else:
                    self.graph.vp.small_component[v] = 0
            print("{} vertices are in the small components.".format(
                num_vertices_in_small_components))

            if purge is True:
                print('Filtering out those vertices and their edges belonging '
                      'to the small components...')
                # Set the filter to get only vertices NOT belonging to a small
                # component.
                self.graph.set_vertex_filter(
                    self.graph.vp.small_component.t(lambda x: 1-x))
                # Purge filtered out vertices and edges from the graph:
                self.graph.purge_vertices()
                # Update graph's dictionary coordinates_to_vertex_index:
                self.update_coordinates_to_vertex_index()

            # Remove the property used for the filtering that is no longer true:
            del self.graph.vertex_properties["small_component"]

        t_end = time.time()
        duration = t_end - t_begin
        minutes, seconds = divmod(duration, 60)
        print('Finding small components took: {} min {} s'.format(
            minutes, seconds))

    # * The following TriangleGraph methods are implementing with adaptations
    # the first step of normal vector voting algorithm of Page et al., 2002. *

    def collect_normal_votes(self, vertex_v_ind, g_max, a_max, sigma,
                             full_dist_map=None):
        """
        For a vertex v, collects the normal votes of all triangles within its
        geodesic neighborhood and calculates the weighted covariance matrix sum
        V_v.

        Implements equations (6), illustrated in figure 6(b), (7) and (8) from
        the paper of Page et al., 2002.

        More precisely, a normal vote n_i of each triangle i (whose centroid c_i
        is lying within the geodesic neighborhood of vertex v) is calculated
        using the normal n assigned to the triangle i. Then, each vote is
        represented by a covariance matrix V_i and votes are collected as a
        weighted matrix sum V_v, where each vote is weighted depending on the
        area of triangle i and the geodesic distance of its centroid c_i from v.

        Here, c_i and v are both centroids of triangles (v is a triangle vertex
        in Page's approach), which are vertices of TriangleGraph generated from
        the triangle surface.

        Args:
            vertex_v_ind (int): index of the vertex v in the surface
                triangle-graph for which the votes are collected
            g_max (float): the maximal geodesic distance in units of the graph
            a_max (float): the area of the largest triangle in the surface
                triangle-graph
            sigma (float): sigma, defined as 3*sigma = g_max, so that votes
                beyond the neighborhood can be ignored
            full_dist_map (graph_tool.PropertyMap, optional): the full distance
                map for the whole graph; if None, a local distance map is
                calculated for this vertex (default)

        Returns:
            - number of geodesic neighbors of vertex v
            - the 3x3 symmetric matrix V_v (numpy.ndarray)
        """
        # To spare function referencing every time in the following for loop:
        vertex = self.graph.vertex
        normal = self.graph.vp.normal
        array = np.array
        xyz = self.graph.vp.xyz
        sqrt = math.sqrt
        dot = np.dot
        outer = np.multiply.outer
        area = self.graph.vp.area
        exp = math.exp

        # Get the coordinates of vertex v (as numpy array):
        vertex_v = self.graph.vertex(vertex_v_ind)
        v = xyz[vertex_v]
        v = array(v)

        # Find the neighboring vertices of vertex v to be returned:
        if vertex_v_ind == 0:
            print("Calling find_geodesic_neighbors")
        neighbor_idx_to_dist = self.find_geodesic_neighbors(
            vertex_v, g_max, full_dist_map=full_dist_map)
        try:
            assert len(neighbor_idx_to_dist) > 0
        except AssertionError:
            print("\nWarning: the vertex v = {} has 0 neighbors. It will be "
                  "ignored later.".format(v))
            # return a placeholder instead of V_v
            return 0, np.zeros(shape=(3, 3))

        # Initialize the weighted matrix sum of all votes for vertex v to be
        # calculated and returned:
        V_v = np.zeros(shape=(3, 3))

        # Let each of the neighboring vertices to cast a vote on vertex v:
        for idx_c_i in list(neighbor_idx_to_dist.keys()):
            # Get neighboring vertex c_i and its coordinates (as numpy array):
            vertex_c_i = vertex(idx_c_i)
            c_i = xyz[vertex_c_i]
            c_i = array(c_i)

            # Calculate the normal vote n_i of c_i on v:
            n = normal[vertex_c_i]
            n = array(n)

            vc_i = c_i - v
            vc_i_len = sqrt(dot(vc_i, vc_i))
            vc_i_norm = vc_i / vc_i_len

            # theta_i is the angle between the vectors n and vc_i
            cos_theta_i = - (dot(n, vc_i)) / vc_i_len

            n_i = n + 2 * cos_theta_i * vc_i_norm

            # Covariance matrix containing one vote of c_i on v:
            V_i = outer(n_i, n_i)

            # Calculate the weight depending on the area of the neighboring
            # triangle i, a_i, and the geodesic distance to the neighboring
            # vertex c_i from vertex v, g_i:
            a_i = area[vertex_c_i]
            g_i = neighbor_idx_to_dist[idx_c_i]
            w_i = a_i / a_max * exp(- g_i / sigma)

            # Weigh V_i and add it to the weighted matrix sum:
            V_v += w_i * V_i

        return len(neighbor_idx_to_dist), V_v
