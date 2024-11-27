import copy
import itertools
import time

import numpy as np
import plotly.graph_objects as go

from scipy.spatial.transform import Rotation as R
from plotly.subplots import make_subplots

from sym_representation import *


def set_obj_visu_params_tless(sym_class, colours='RGB'):
    # this function defines the colours and shapes of the polygons used in the visualization
    p_z_color = [0, 0, 255]
    n_z_color = [0, 0, 128]
    p_x_color = [255, 0, 0]
    p_xxy_color = [191, 0, 0]
    p_xy_color = [127, 223, 0]
    p_xyy_color = [64, 239, 0]
    p_y_color = [0, 255, 0]
    p_yy_n_x_color = [32, 191, 0]
    p_y_n_x_color = [64, 127, 64]
    p_y_n_xx_color = [96, 63, 128]
    n_x_color = [128, 0, 0]
    n_xxy_color = [96, 32, 128]
    n_xy_color = [64, 64, 128]
    n_xyy_color = [32, 96, 128]
    n_y_color = [0, 128, 0]
    n_yy_p_x_color = [64, 96, 128]
    n_y_p_x_color = [128, 64, 64]
    n_y_p_xx_color = [192, 32, 0]

    if colours == 'RGB':
        face_colours = [n_y_color, n_y_color, n_z_color, n_z_color, p_z_color, p_z_color, p_y_color, p_y_color, n_x_color, n_x_color, p_x_color, p_x_color]
    else:
        face_colours = [127, 127, 127]

    if sym_class == 'I':
        edges = np.array([[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2, 15, 8, 8, 8, 12, 12, 14, 14, 12, 8, 11, 10, 23, 16, 16, 16, 20, 20, 22, 22, 20, 16, 19, 18],
                                       [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3, 11, 12, 9, 10, 13, 14, 13, 10, 8, 9, 14, 11, 19, 20, 17, 18, 21, 22, 21, 18, 16, 17, 22, 19],
                                       [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6, 8, 15, 10, 11, 14, 15, 9, 9, 13, 13, 15, 14, 16, 23, 18, 19, 22, 23, 17, 17, 21, 21, 23, 22]])

        vertices = np.array([[-1, -1, 1, 1, -1, -1, 1, 1, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5],
                                           [-1, 1, 1, -1, -1, 1, 1, -1, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0, 2, 2, 0, 0, 2, 2, 0],
                                           [-1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5]], dtype=np.float32)

        vertices[0] *= 0.25
        vertices[1] *= 0.5
        vertices[2] *= 0.75

    elif sym_class == 'II':
        edges = np.array([[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2, 15, 8, 8, 8, 12, 12, 14, 14, 12, 8, 11, 10],
                                       [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3, 11, 12, 9, 10, 13, 14, 13, 10, 8, 9, 14, 11],
                                       [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6, 8, 15, 10, 11, 14, 15, 9, 9, 13, 13, 15, 14]])

        vertices = np.array([[-1, -1, 1, 1, -1, -1, 1, 1, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5],
                                           [-1, 1, 1, -1, -1, 1, 1, -1, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5],
                                           [-1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2]], dtype=np.float32)

        vertices[0] *= 0.25
        vertices[1] *= 0.5
        vertices[2] *= 0.75

    elif sym_class == 'III':
        edges = np.array([[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2, 15, 8, 8, 8, 12, 12, 14, 14, 12, 8, 11, 10],
                                       [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3, 11, 12, 9, 10, 13, 14, 13, 10, 8, 9, 14, 11],
                                       [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6, 8, 15, 10, 11, 14, 15, 9, 9, 13, 13, 15, 14]])

        vertices = np.array([[-1, -1, 1, 1, -1, -1, 1, 1, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5],
                                           [-1, 1, 1, -1, -1, 1, 1, -1, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5],
                                           [-1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2]], dtype=np.float32)

        vertices[0] *= 0.4
        vertices[1] *= 0.4
        vertices[2] *= 0.75

    elif sym_class == 'IV':
        num_points = 16
        height = 3
        radius = 1

        # Apex of the cone
        vertices = []
        vertices.append([0, 0, height / 2])

        # Base of the cone
        # Base vertices
        angle_step = 2 * math.pi / num_points
        for i in range(num_points):
            angle = i * angle_step
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = -height / 2
            vertices.append([x, y, z])

        # Center of the base
        vertices.append([0, 0, -height / 2])
        vertices = np.array(vertices).T

        # Base faces
        faces = []
        for i in range(num_points):
            v0 = i + 1
            v1 = (i + 1) % num_points + 1
            v2 = num_points + 1  # Center of the base
            faces.append([v0, v1, v2])

        # Side faces of the cone
        for i in range(num_points):
            v0 = i + 1
            v1 = (i + 1) % num_points + 1
            v2 = 0  # Apex of the cone
            faces.append([v0, v2, v1])

        edges = np.array(faces).T

        face_colours = [p_x_color, p_xxy_color, p_xy_color, p_xyy_color, p_y_color, p_yy_n_x_color, p_y_n_x_color, p_y_n_xx_color, n_x_color, n_xxy_color, n_xy_color, n_xyy_color, n_y_color, n_yy_p_x_color,
                                   n_y_p_x_color, n_y_p_xx_color, p_x_color, p_xxy_color, p_xy_color, p_xyy_color, p_y_color, p_yy_n_x_color, p_y_n_x_color, p_y_n_xx_color, n_x_color, n_xxy_color, n_xy_color, n_xyy_color, n_y_color,
                                   n_yy_p_x_color, n_y_p_x_color, n_y_p_xx_color]

    elif sym_class == 'V':
        edges = np.array([[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2, 15, 8, 8, 8, 12, 12, 14, 14, 12, 8, 11, 10],
                                        [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3, 11, 12, 9, 10, 13, 14, 13, 10, 8, 9, 14, 11],
                                        [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6, 8, 15, 10, 11, 14, 15, 9, 9, 13, 13, 15, 14]])

        vertices = np.array([[-1, -1, 1, 1, -1, -1, 1, 1, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5],
                                           [-1, 1, 1, -1, -1, 1, 1, -1, 0, 2, 2, 0, 0, 2, 2, 0],
                                           [-1, -1, -1, -1, 1, 1, 1, 1, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5]], dtype=np.float32)

        vertices[0] *= 0.25
        vertices[1] *= 0.5
        vertices[2] *= 0.75
    else:
        raise ValueError('Unrecognized sym_class')

    return vertices, edges, face_colours


def map_to_sym_representation(rots, sym_cls):
    # This maps to our representation
    s_a_ = []
    c_a_ = []
    s_b_ = []
    c_b_ = []
    s_g_ = []
    c_g_ = []

    idx = 0
    for rot in rots:
        alpha = rot[0]
        beta = rot[1]
        gamma = rot[2]
        rot_sym = sym_aware_rotation(alpha, beta, gamma, sym_cls)
        s_a_.append(rot_sym[0][0])
        c_a_.append(rot_sym[1][0])
        s_b_.append(rot_sym[0][1])
        c_b_.append(rot_sym[1][1])
        s_g_.append(rot_sym[0][2])
        c_g_.append(rot_sym[1][2])
        idx += 1

    s_a_ = np.array(s_a_)
    c_a_ = np.array(c_a_)
    s_b_ = np.array(s_b_)
    c_b_ = np.array(c_b_)
    s_g_ = np.array(s_g_)
    c_g_ = np.array(c_g_)

    return s_a_, c_a_, s_b_, c_b_, s_g_, c_g_


def plot_mapping_tless(sym_cls=None, colours='RGB'):
    ### SETUP ###
    sym_v = TLESS_SYM_CLASSES[sym_cls]['sym_v']
    vertices, edges, face_colours = set_obj_visu_params_tless(sym_cls, colours)
    vertices /= 5
    axis_outer_bound_offset = 0.3

    s_titles = [
                    rf'$s_{{{sym_cls}, \alpha}}$',
                    rf'$s_{{{sym_cls}, \beta}}$',
                    rf'$s_{{{sym_cls}, \gamma}}\text{{ (values are only colour-coded)}}$',
                    rf'$c_{{{sym_cls}, \alpha}}$',
                    rf'$c_{{{sym_cls}, \beta}}$',
                    rf'$c_{{{sym_cls}, \gamma}}\text{{ (values are only colour-coded)}}$'
    ]

    fig = make_subplots(rows=2, cols=3, specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}], [{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]], subplot_titles=s_titles, horizontal_spacing=0.01, vertical_spacing=0.02)

    a1 = np.deg2rad(np.arange(5, 90, 10))
    a2 = np.deg2rad(np.arange(275, 360, 10)) if sym_cls != 'V' else a1
    b1 = np.deg2rad(np.array([0]))
    b2 = np.deg2rad(np.array([180])) if sym_cls != 'V' else b1
    g1 = np.deg2rad(np.arange(0, 360, 5))
    g2 = np.deg2rad(np.arange(0, 360, 5))

    t1 = [(a, b, g) for a, b, g in itertools.product(a1, b1, g1)]
    t2 = [(a, b, g) for a, b, g in itertools.product(a2, b2, g2)]

    s_a_1, c_a_1, s_b_1, c_b_1, s_g_1, c_g_1 = map_to_sym_representation(t1, sym_cls=sym_cls)
    s_a_2, c_a_2, s_b_2, c_b_2, s_g_2, c_g_2 = map_to_sym_representation(t2, sym_cls=sym_cls)

    x_grid_1, y_grid_1, z_grid_1 = np.meshgrid(a1, b1, g1, indexing='ij')
    z1_flat = z_grid_1.flatten()
    y1_flat = y_grid_1.flatten()
    x1_flat = x_grid_1.flatten()
    
    x_grid_2, y_grid_2, z_grid_2 = np.meshgrid(a2, b2, g2, indexing='ij')
    z2_flat = z_grid_2.flatten()
    y2_flat = y_grid_2.flatten()
    x2_flat = x_grid_2.flatten()
    

    ### ALPHA ###
    scatter_s_1 = go.Scatter3d(x=x1_flat, y=y1_flat, z=z1_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=s_a_1, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter_s_1, row=1, col=1)
    scatter_s_2= go.Scatter3d(x=x2_flat, y=y2_flat, z=z2_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=s_a_2, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter_s_2, row=1, col=1)
    scatter_c_1 = go.Scatter3d(x=x1_flat, y=y1_flat, z=z1_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=c_a_1, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter_c_1, row=2, col=1)
    scatter_c_2= go.Scatter3d(x=x2_flat, y=y2_flat, z=z2_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=c_a_2, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter_c_2, row=2, col=1)

    ### BETA ###
    scatter_s_1 = go.Scatter3d(x=x1_flat, y=y1_flat, z=z1_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=s_b_1, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter_s_1, row=1, col=2)
    scatter_s_2 = go.Scatter3d(x=x2_flat, y=y2_flat, z=z2_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=s_b_2, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter_s_2, row=1, col=2)
    scatter_c_1 = go.Scatter3d(x=x1_flat, y=y1_flat, z=z1_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=c_b_1, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter_c_1, row=2, col=2)
    scatter_c_2 = go.Scatter3d(x=x2_flat, y=y2_flat, z=z2_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=c_b_2, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter_c_2, row=2, col=2)

    ### GAMMA ###
    scatter_s_1 = go.Scatter3d(x=x1_flat, y=y1_flat, z=z1_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=s_g_1, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter_s_1, row=1, col=3)
    scatter_s_2 = go.Scatter3d(x=x2_flat, y=y2_flat, z=z2_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=s_g_2, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter_s_2, row=1, col=3)
    scatter_c_1 = go.Scatter3d(x=x1_flat, y=y1_flat, z=z1_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=c_g_1, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter_c_1, row=2, col=3)
    scatter_c_2 = go.Scatter3d(x=x2_flat, y=y2_flat, z=z2_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=c_g_2, colorscale='viridis', cmin=-1, cmax=1, colorbar=dict(thickness=20)))
    fig.add_trace(scatter_c_2, row=2, col=3)

    for j in range(1, 4):
        for i in range(1, 3):
            for g_idx, gamma in enumerate(g1):
                if ((g_idx % 6) != 0) and (g_idx != (len(g2) - 1)):
                    continue
                for b_idx, beta in enumerate(b1):
                    for a_idx, alpha in enumerate(a1):
                        if (a_idx % 4 != 0):
                            continue
                        vertex = vertices.copy()
                        r = R.from_euler('XYZ', [alpha, beta, gamma])
                        mat = r.as_matrix()
                        # apply transformation
                        vertex = np.dot(mat, vertex)
                        vertex[0] += alpha
                        vertex[1] += beta
                        vertex[2] += gamma
                        mesh = go.Mesh3d(x=vertex[0, :], y=vertex[1, :], z=vertex[2, :], i=edges[0], j=edges[1], k=edges[2], opacity=1.0, facecolor=face_colours, flatshading=True)
                        fig.add_trace(mesh, row=i, col=j)

        for i in range(1, 3):
            for g_idx, gamma in enumerate(g2):
                if ((g_idx % 6) != 0) and (g_idx != (len(g2) - 1)):
                    continue
                for b_idx, beta in enumerate(b2):
                    for a_idx, alpha in enumerate(a2):
                        if (a_idx % 4 != 0):
                            continue
                        vertex = vertices.copy()
                        r = R.from_euler('XYZ', [alpha, beta, gamma])
                        mat = r.as_matrix()
                        # apply transformation
                        vertex = np.dot(mat, vertex)
                        vertex[0] += alpha
                        vertex[1] += beta
                        vertex[2] += gamma
                        mesh = go.Mesh3d(x=vertex[0, :], y=vertex[1, :], z=vertex[2, :], i=edges[0], j=edges[1], k=edges[2], opacity=1.0, facecolor=face_colours, flatshading=True)
                        fig.add_trace(mesh, row=i, col=j)

            fig.update_scenes(row=i, col=j, xaxis_title_text='alpha', yaxis_title_text='beta', zaxis_title_text='gamma',
                              xaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset],
                              yaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset],
                              zaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset], aspectmode='cube')

    fig['layout']['title'] = rf'$\text{{Plots of the six parameters that make up our representation }}\mathcal{{S}}_{{{sym_cls}}}, \kappa_{{{sym_cls}}}={sym_v} \text{{ ("XYZ" intrinsic rotation order, TLESS rotation subspace, angles in radians)}}$'
    #fig_3['layout']['dragmode'] = 'orbit'

    #fig.show()
    fig.write_html(f"{sym_cls}_TLESS.html", include_mathjax='cdn')


def plot_mapping_whole(sym_cls=None, colours='RGB'):
    ### SETUP ###
    sym_v = TLESS_SYM_CLASSES[sym_cls]['sym_v']
    vertices, edges, face_colours = set_obj_visu_params_tless(sym_cls, colours)
    vertices /= 5
    axis_outer_bound_offset = 0.3

    s_titles = [
        rf'$s_{{{sym_cls}, \alpha}}$',
        rf'$s_{{{sym_cls}, \beta}}$',
        rf'$s_{{{sym_cls}, \gamma}}\text{{ (values are only colour-coded)}}$',
        rf'$c_{{{sym_cls}, \alpha}}$',
        rf'$c_{{{sym_cls}, \beta}}$',
        rf'$c_{{{sym_cls}, \gamma}}\text{{ (values are only colour-coded)}}$'
    ]

    fig = make_subplots(rows=2, cols=3, specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}], [{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]], subplot_titles=s_titles, horizontal_spacing=0.01, vertical_spacing=0.02)

    a = np.deg2rad(np.arange(0, 360, 20))
    b = np.deg2rad(np.arange(0, 360, 20))
    g = np.deg2rad(np.arange(0, 360, 20))

    t = [(a, b, g) for a, b, g in itertools.product(a, b, g)]
    s_a, c_a, s_b, c_b, s_g, c_g = map_to_sym_representation(t, sym_cls=sym_cls)
    
    x_grid, y_grid, z_grid = np.meshgrid(a, b, g, indexing='ij')
    z_flat = z_grid.flatten()
    y_flat = y_grid.flatten()
    x_flat = x_grid.flatten()

    ### ALPHA ###
    scatter_s = go.Scatter3d(x=x_flat, y=y_flat, z=z_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=s_a, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter_s, row=1, col=1)
    scatter_c = go.Scatter3d(x=x_flat, y=y_flat, z=z_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=c_a, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter_c, row=2, col=1)

    ### BETA ###
    scatter_s = go.Scatter3d(x=x_flat, y=y_flat, z=z_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=s_b, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter_s, row=1, col=2)
    scatter_c = go.Scatter3d(x=x_flat, y=y_flat, z=z_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=c_b, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter_c, row=2, col=2)

    ### GAMMA ###
    scatter_s = go.Scatter3d(x=x_flat, y=y_flat, z=z_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=s_g, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter_s, row=1, col=3)
    scatter_c = go.Scatter3d(x=x_flat, y=y_flat, z=z_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=c_g, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter_c, row=2, col=3)

    for j in range(1, 4):
        for i in range(1, 3):
            for g_idx, gamma in enumerate(g):
                if ((g_idx % 3) != 0) and (g_idx != (len(g) - 1)):
                    continue
                for b_idx, beta in enumerate(b):
                    if (b_idx % 3 != 0) and (b_idx != (len(b) - 1)):
                        continue
                    for a_idx, alpha in enumerate(a):
                        if (a_idx % 3 != 0) and (a_idx != (len(a) - 1)):
                            continue
                        vertex = vertices.copy()
                        r = R.from_euler('XYZ', [alpha, beta, gamma])
                        mat = r.as_matrix()
                        # apply transformation
                        vertex = np.dot(mat, vertex)
                        vertex[0] += alpha
                        vertex[1] += beta
                        vertex[2] += gamma
                        mesh = go.Mesh3d(x=vertex[0, :], y=vertex[1, :], z=vertex[2, :], i=edges[0], j=edges[1], k=edges[2], opacity=1.0, facecolor=face_colours, flatshading=True)
                        fig.add_trace(mesh, row=i, col=j)

            fig.update_scenes(row=i, col=j, xaxis_title_text='alpha', yaxis_title_text='beta', zaxis_title_text='gamma',
                              xaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset],
                              yaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset],
                              zaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset], aspectmode='cube')
    fig['layout']['title'] = rf'$\text{{Plots of the six parameters that make up our representation }}\mathcal{{S}}_{{{sym_cls}}}, \kappa_{{{sym_cls}}}={sym_v} \text{{ ("XYZ" intrinsic rotation order, SO(3) rotation space, angles in radians)}}$'
    # fig_3['layout']['dragmode'] = 'orbit'

    #fig.show()
    fig.write_html(f"{sym_cls}_SO3.html", include_mathjax='cdn')


if __name__ == '__main__':
    sym_classes = ['I', 'II', 'III', 'IV']
    colours = 'RGB'
    subspaces = ['TLESS', 'SO3']

    for subspace in subspaces:
        for sym_cls in sym_classes:
            if subspace == 'TLESS':
                plot_mapping_tless(sym_cls=sym_cls, colours=colours)
                print(f'Plots of the six parameters that make up our representation S_{sym_cls}, with kappa_{sym_cls}, TLESS rotation space')
            elif subspace == 'SO3':
                plot_mapping_whole(sym_cls=sym_cls, colours=colours)
                print(f'Plots of the six parameters that make up our representation S_{sym_cls}, with kappa_{sym_cls}, SO(3) rotation space')
            else:
                raise NotImplementedError
            time.sleep(60)

