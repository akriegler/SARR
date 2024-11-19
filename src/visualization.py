import copy

import numpy as np
import plotly.graph_objects as go

from scipy.spatial.transform import Rotation as R
from plotly.subplots import make_subplots

from sym_representation import *


def set_obj_visu_params_tless(sym_class, colours='RGB'):
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
        face_colours = [n_y_color, n_y_color, n_z_color, n_z_color, p_z_color, p_z_color, p_y_color, p_y_color,
                        n_x_color, n_x_color, p_x_color, p_x_color]
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

        vertices = []
        angle_step = 2 * math.pi / num_points

        # Apex of the cone
        vertices.append([0, 0, height / 2])

        # Base vertices
        for i in range(num_points):
            angle = i * angle_step
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = -height / 2
            vertices.append([x, y, z])

        # Center of the base
        vertices.append([0, 0, -height / 2])

        vertices = np.array(vertices).T

        faces = []

        # Base faces
        for i in range(num_points):
            v0 = i + 1
            v1 = (i + 1) % num_points + 1
            v2 = num_points + 1  # Center of the base
            faces.append([v0, v1, v2])

        # Side faces
        for i in range(num_points):
            v0 = i + 1
            v1 = (i + 1) % num_points + 1
            v2 = 0  # Apex of the cone
            faces.append([v0, v2, v1])

        edges = np.array(faces).T

        face_colours = [p_x_color, p_xxy_color, p_xy_color, p_xyy_color, p_y_color, p_yy_n_x_color, p_y_n_x_color,
                            p_y_n_xx_color, n_x_color, n_xxy_color, n_xy_color, n_xyy_color, n_y_color, n_yy_p_x_color,
                            n_y_p_x_color, n_y_p_xx_color, p_x_color, p_xxy_color, p_xy_color, p_xyy_color, p_y_color,
                            p_yy_n_x_color, p_y_n_x_color,
                            p_y_n_xx_color, n_x_color, n_xxy_color, n_xy_color, n_xyy_color, n_y_color, n_yy_p_x_color,
                            n_y_p_x_color, n_y_p_xx_color]

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

    return vertices, edges, face_colours


def map(alphas, betas, gammas, sym_cls):
    s_a_ = []
    c_a_ = []
    s_b_ = []
    c_b_ = []
    s_g_ = []
    c_g_ = []

    idx = 0
    for gamma in gammas:
        for beta in betas:
            for alpha in alphas:
                rot_sym = sym_aware_rotation(alpha, beta, gamma, sym_cls, clamp=True)
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


def plot_mapping_tless(sym_cls=None, steps=50, visu_steps=5, colours='RGB'):

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

    a1 = np.linspace(0.0, np.pi / 2, steps // 2)
    a2 = np.linspace(np.pi, 6/4 * np.pi, steps // 2)

    a = np.concatenate((a1, a2), axis=0)
    b = np.zeros_like(a)
    g = np.linspace(0, 2 * np.pi, steps)

    s_a_, c_a_, s_b_, c_b_, s_g_, c_g_ = map(alphas=a, betas=b, gammas=g, sym_cls=sym_cls)

    ### ALPHA ###
    scatter1 = go.Scatter3d(x=a, y=np.zeros_like(a), z=s_a_[:len(a)], mode='markers', marker=dict(size=3, opacity=1.0, color=s_a_[:len(a)], colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter1, row=1, col=1)
    scatter2 = go.Scatter3d(x=a, y=np.zeros_like(a), z=c_a_[:len(a)], mode='markers', marker=dict(size=3, opacity=1.0, color=c_a_[:len(a)], colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter2, row=2, col=1)

    a1 = np.concatenate((np.linspace(0.0, np.pi / 2, steps // 2)[::visu_steps], np.array([a1[-1]])))
    a2 = np.concatenate((np.linspace(np.pi, 6/4 * np.pi, steps // 2)[::visu_steps], np.array([a2[-1]])))
    a_angles = np.concatenate((a1, a2), axis=0)

    s_a_1 = np.concatenate((s_a_[:len(a) // 2:visu_steps], np.array([s_a_[17]])))
    s_a_2 = np.concatenate((s_a_[len(a)//2:len(a):visu_steps], np.array([s_a_[35]])))
    s_a_ = np.concatenate((s_a_1, s_a_2), axis=0)

    c_a_1 = np.concatenate((c_a_[:len(a) // 2:visu_steps], np.array([c_a_[17]])))
    c_a_2 = np.concatenate((c_a_[len(a)//2:len(a):visu_steps], np.array([c_a_[35]])))
    c_a_ = np.concatenate((c_a_1, c_a_2), axis=0)

    for i in range(1, 3):
        for a_idx, alpha in enumerate(a_angles):
            vertex = copy.deepcopy(vertices)

            r = R.from_euler('X', alpha)
            mat = r.as_matrix()

            # apply transformation
            vertex = np.dot(mat, vertex)
            vertex[2] += (s_a_[a_idx] if i == 1 else c_a_[a_idx])
            vertex[0] += alpha

            mesh = go.Mesh3d(x=vertex[0, :], y=vertex[1, :], z=vertex[2, :], i=edges[0], j=edges[1], k=edges[2], opacity=1.0, facecolor=face_colours, flatshading=True)
            fig.add_trace(mesh, row=i, col=1)

    # Update layout and axis labels
    fig.update_scenes(row=1, col=1, yaxis_title_text='-', zaxis_title_text=f's_({sym_cls}, alpha)', xaxis_title_text='alpha', xaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset],
                      yaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset], zaxis_range=[-np.pi - axis_outer_bound_offset, np.pi + axis_outer_bound_offset], aspectmode='cube')
    fig.update_scenes(row=2, col=1, yaxis_title_text='-', zaxis_title_text=f'c_({sym_cls}, alpha)', xaxis_title_text='alpha', xaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset],
                      yaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset], zaxis_range=[-np.pi - axis_outer_bound_offset, np.pi + axis_outer_bound_offset], aspectmode='cube')

    ### BETA ###
    b_angles = np.linspace(0.0, 0.0001, steps + 4)[::visu_steps]

    b_y_m, b_x_m = np.meshgrid(b_angles, a_angles)
    b_x_flat = b_x_m.flatten()
    b_y_flat = b_y_m.flatten()

    selected_indices = [
        block_start + i
        for block_start in [*range(0, 1296, 108), 648, 1296]
        for i in sorted([j - 1 for j in range(1, 36, 3)] + [17, 35])
    ]

    s_b_ = s_b_[selected_indices]
    c_b_ = c_b_[selected_indices]

    scatter3 = go.Scatter3d(x=b_x_flat, y=b_y_flat, z=s_b_, mode='markers', marker=dict(size=3, opacity=1.0, color=s_b_, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter3, row=1, col=2)

    scatter4 = go.Scatter3d(x=b_x_flat, y=b_y_flat, z=c_b_, mode='markers', marker=dict(size=3, opacity=1.0, color=c_b_, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter4, row=2, col=2)

    for i in range(1, 3):
        for b_idx, beta in enumerate(b_angles):
            for a_idx, alpha in enumerate(a_angles):
                vertex = vertices.copy()
                r = R.from_euler('XY', [alpha, beta])
                mat = r.as_matrix()

                # apply transformation
                vertex = np.dot(mat, vertex)
                vertex[0] += alpha
                vertex[1] += beta
                vertex[2] += (s_b_[a_idx + 14 * b_idx] if i == 1 else c_b_[a_idx + 14 * b_idx])
                mesh = go.Mesh3d(x=vertex[0, :], y=vertex[1, :], z=vertex[2, :], i=edges[0], j=edges[1], k=edges[2], opacity=1.0, facecolor=face_colours, flatshading=True)
                fig.add_trace(mesh, row=i, col=2)

    # Update layout and axis labels
    fig.update_scenes(row=1, col=2, zaxis_title_text=f's_{sym_cls}, beta', yaxis_title_text='beta', xaxis_title_text='alpha',
                      zaxis_range=[-np.pi - axis_outer_bound_offset, np.pi + axis_outer_bound_offset], yaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset], xaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset], aspectmode='cube')
    fig.update_scenes(row=2, col=2, zaxis_title_text=f'c_{sym_cls}, beta', yaxis_title_text='beta', xaxis_title_text='alpha',
                      zaxis_range=[-np.pi - axis_outer_bound_offset, np.pi + axis_outer_bound_offset], yaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset], xaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset], aspectmode='cube')

    ### GAMMA ###
    g_angles = np.linspace(0.0, 2 * np.pi, steps + 4)[::visu_steps]
    g_z_m, g_y_m, g_x_m = np.meshgrid(g_angles, b_angles, a_angles)
    g_x_flat = g_x_m.flatten()
    g_y_flat = g_y_m.flatten()
    g_z_flat = g_z_m.flatten()

    selected_blocks = [i for i in range(0, 36, 3)] + [17, 35]  # Every 3rd slice, plus 18th and 36th
    selected_rows = [i for i in range(0, 36, 3)] + [17, 35]  # Every 3rd row, plus 18th and 36th
    selected_slices = [i for i in range(0, 36, 3)] + [17, 35]  # Every 3rd element in a row, plus 18th and 36th
    # To get the flat indices
    indices = [
        block * 1296 + row * 36 + slice_
        for block in selected_blocks
        for row in selected_rows
        for slice_ in selected_slices
    ]
    indices.sort()
    s_g_ = s_g_[indices]
    c_g_ = c_g_[indices]

    scatter = go.Scatter3d(x=g_x_flat, y=g_y_flat, z=g_z_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=s_g_, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter, row=1, col=3)

    scatter = go.Scatter3d(x=g_x_flat, y=g_y_flat, z=g_z_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=c_g_, colorscale='viridis', cmin=-1, cmax=1, colorbar=dict(thickness=20)))
    fig.add_trace(scatter, row=2, col=3)

    for i in range(1, 3):
        for g_idx, gamma in enumerate(g_angles):
            for b_idx, beta in enumerate(b_angles):
                for a_idx, alpha in enumerate(a_angles):
                    vertex = vertices.copy()
                    r = R.from_euler('XYZ', [alpha, beta, gamma])
                    mat = r.as_matrix()
                    # apply transformation
                    vertex = np.dot(mat, vertex)
                    vertex[0] += alpha
                    vertex[1] += beta
                    vertex[2] += gamma
                    mesh = go.Mesh3d(x=vertex[0, :], y=vertex[1, :], z=vertex[2, :], i=edges[0], j=edges[1], k=edges[2], opacity=1.0, facecolor=face_colours, flatshading=True)
                    fig.add_trace(mesh, row=i, col=3)

    fig.update_scenes(row=1, col=3, xaxis_title_text='alpha', yaxis_title_text='beta', zaxis_title_text='gamma',
                        xaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset],
                        yaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset],
                        zaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset], aspectmode='cube')
    fig.update_scenes(row=2, col=3, xaxis_title_text='alpha', yaxis_title_text='beta', zaxis_title_text='gamma',
                        xaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset],
                        yaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset],
                        zaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset], aspectmode='cube')
    fig['layout']['title'] = rf'$\text{{Plots of the six parameters that make up our representation }}\mathcal{{S}}_{{{sym_cls}}}, \kappa_{{{sym_cls}}}={sym_v} \text{{ ("XYZ" intrinsic rotation order, TLESS rotation subspace, angles in radians)}}$'
    #fig_3['layout']['dragmode'] = 'orbit'

    fig.show()
    #fig.write_html(f"{sym_cls}.html", include_mathjax='cdn')


def plot_mapping_whole(sym_cls=None, steps=50, visu_steps=5, colours='RGB'):
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

    a = np.linspace(0.0, 2*np.pi, steps)
    b = np.linspace(0.0, 2*np.pi, steps)
    g = np.linspace(0.0, 2*np.pi, steps)

    s_a_, c_a_, s_b_, c_b_, s_g_, c_g_ = map(alphas=a, betas=b, gammas=g, sym_cls=sym_cls)

    ### ALPHA ###
    scatter1 = go.Scatter3d(x=a, y=np.zeros_like(a), z=s_a_[:len(a)], mode='markers', marker=dict(size=3, opacity=1.0, color=s_a_[:len(a)], colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter1, row=1, col=1)
    scatter2 = go.Scatter3d(x=a, y=np.zeros_like(a), z=c_a_[:len(a)], mode='markers', marker=dict(size=3, opacity=1.0, color=c_a_[:len(a)], colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter2, row=2, col=1)

    s_a_ = np.array([s_a_[i] for i in sorted([j - 1 for j in range(1, 36, 3)] + [17, 35])])
    c_a_ = np.array([c_a_[i] for i in sorted([j - 1 for j in range(1, 36, 3)] + [17, 35])])

    angles = a[::visu_steps]

    for i in range(1, 3):
        for a_idx, alpha in enumerate(angles):
            vertex = copy.deepcopy(vertices)

            r = R.from_euler('X', alpha)
            mat = r.as_matrix()
            # apply transformation
            vertex = np.dot(mat, vertex)
            vertex[2] += (s_a_[a_idx] if i == 1 else c_a_[a_idx])
            vertex[0] += alpha

            mesh = go.Mesh3d(x=vertex[0, :], y=vertex[1, :], z=vertex[2, :], i=edges[0], j=edges[1], k=edges[2], opacity=1.0, facecolor=face_colours, flatshading=True)
            fig.add_trace(mesh, row=i, col=1)


    # Update layout and axis labels
    fig.update_scenes(row=1, col=1, yaxis_title_text='-',
                      zaxis_title_text=f's_({sym_cls}, alpha)',
                      xaxis_title_text='alpha', xaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset],
                      yaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset],
                      zaxis_range=[-np.pi - axis_outer_bound_offset, np.pi + axis_outer_bound_offset], aspectmode='cube')
    fig.update_scenes(row=2, col=1, yaxis_title_text='-',
                      zaxis_title_text=f'c_({sym_cls}, alpha)',
                      xaxis_title_text='alpha', xaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset],
                      yaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset],
                      zaxis_range=[-np.pi - axis_outer_bound_offset, np.pi + axis_outer_bound_offset], aspectmode='cube')

    fig.show()

    hugo

    ######## BETA ########
    b = np.linspace(0, 2*np.pi, steps + 1)#(0 if sym_v[1] == 10**6 else (2 * np.pi - np.deg2rad(1)) / sym_v[1]), steps + 1)
    #b = np.array([np.pi, np.pi])
    b_factor = np.array([math.cos(ang) for ang in b])

    y_m, x_m = np.meshgrid(b, a_factor)

    s_b_s = map(ang=y_m, sym_v=sym_v[1], c_b=np.ones_like(x_m), var='s')

    y_m, x_m = np.meshgrid(b, a)

    x_flat = x_m.flatten()
    y_flat = y_m.flatten()
    s_b_s_flat = s_b_s.flatten()

    scatter3 = go.Scatter3d(x=x_flat, y=y_flat, z=s_b_s_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=s_b_s_flat, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter3, row=1, col=2)

    y_m, x_m = np.meshgrid(b, a_factor)

    c_b_s = map(ang=y_m, sym_v=sym_v[1], var='c')

    y_m, x_m = np.meshgrid(b, a)

    x_flat = x_m.flatten()
    y_flat = y_m.flatten()
    c_b_s_flat = c_b_s.flatten()

    scatter4 = go.Scatter3d(x=x_flat, y=y_flat, z=c_b_s_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=c_b_s_flat, colorscale='viridis', cmin=-1, cmax=1))

    fig.add_trace(scatter4, row=2, col=2)

    num_cuboids = (a.shape[0] // visu_steps) + 1
    angles = np.zeros((3, num_cuboids))

    angles[0, :] = a[::visu_steps]
    angles[1, :] = b[::visu_steps]

    for i in range(1, 3):
        for a_idx, alpha in enumerate(angles[0, :]):
            for b_idx, beta in enumerate(angles[1, :]):
                vertex = vertices.copy()
                r = R.from_euler('XY', [alpha, beta])
                mat = r.as_matrix()

                # apply transformation
                vertex = np.dot(mat, vertex)
                vertex[0] += alpha
                vertex[1] += beta
                vertex[2] += (s_b_s[a_idx * visu_steps, b_idx * visu_steps] if i == 1 else c_b_s[a_idx * visu_steps, b_idx * visu_steps])
                mesh = go.Mesh3d(x=vertex[0, :], y=vertex[1, :], z=vertex[2, :], i=edges[0], j=edges[1], k=edges[2],
                                 opacity=1.0, facecolor=face_colours, flatshading=True)
                fig.add_trace(mesh, row=i, col=2)

    # Update layout and axis labels
    fig.update_scenes(row=1, col=2, zaxis_title_text=f's_{sym_cls}, beta',
                        yaxis_title_text='beta', xaxis_title_text='alpha',
                        zaxis_range=[-np.pi - axis_outer_bound_offset, np.pi + axis_outer_bound_offset],
                        yaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset],
                        xaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset], aspectmode='cube')
    fig.update_scenes(row=2, col=2, zaxis_title_text=f'c_{sym_cls}, beta',
                        yaxis_title_text='beta',
                        xaxis_title_text='alpha',
                        zaxis_range=[-np.pi - axis_outer_bound_offset, np.pi + axis_outer_bound_offset],
                        yaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset],
                        xaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset], aspectmode='cube')
    #fig_2['layout']['dragmode'] = 'orbit'

    ###### GAMMA ######
    g = np.linspace(0, 2*np.pi, steps + 1)
    #a = np.deg2rad(np.array([50, 230]))
    z_m, y_m, x_m = np.meshgrid(g, b_factor, a_factor)
    s_g_s = map(ang=z_m, sym_v=sym_v[2], c_b=y_m if sym_v[1] > 1 else np.ones_like(y_m), var='s')
    c_g_s = map(ang=z_m, sym_v=sym_v[2], var='c')
    s_g_s_flat = s_g_s.flatten()
    c_g_s_flat = c_g_s.flatten()

    z_m, y_m, x_m = np.meshgrid(g, b, a)

    x_flat = x_m.flatten()
    y_flat = y_m.flatten()
    z_flat = z_m.flatten()

    scatter = go.Scatter3d(x=x_flat, y=y_flat, z=z_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=s_g_s_flat, colorscale='viridis', cmin=-1, cmax=1))
    fig.add_trace(scatter, row=1, col=3)

    scatter = go.Scatter3d(x=x_flat, y=y_flat, z=z_flat, mode='markers', marker=dict(size=3, opacity=1.0, color=c_g_s_flat, colorscale='viridis', cmin=-1, cmax=1, colorbar=dict(thickness=20)))
    fig.add_trace(scatter, row=2, col=3)

    angles = np.zeros((3, g.shape[0] // visu_steps + 1))
    angles[2, :] = g[::visu_steps]
    angles[1, :] = b[::visu_steps]
    angles[0, :] = a[::visu_steps]

    for i in range(1, 3):
        for g_idx, gamma in enumerate(angles[2, :]):
            for b_idx, beta in enumerate(angles[1, :]):
                for a_idx, alpha in enumerate(angles[0, :]):
                    vertex = vertices.copy()
                    r = R.from_euler('XYZ', [alpha, beta, gamma])
                    mat = r.as_matrix()
                    # apply transformation
                    vertex = np.dot(mat, vertex)
                    vertex[0] += alpha
                    vertex[1] += beta
                    vertex[2] += gamma
                    mesh = go.Mesh3d(x=vertex[0, :], y=vertex[1, :], z=vertex[2, :], i=edges[0], j=edges[1], k=edges[2],
                                     opacity=1.0, facecolor=face_colours, flatshading=True)
                    fig.add_trace(mesh, row=i, col=3)

    fig.update_scenes(row=1, col=3, xaxis_title_text='alpha', yaxis_title_text='beta', zaxis_title_text='gamma',
                        xaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset],
                        yaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset],
                        zaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset], aspectmode='cube')
    fig.update_scenes(row=2, col=3, xaxis_title_text='alpha', yaxis_title_text='beta', zaxis_title_text='gamma',
                        xaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset],
                        yaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset],
                        zaxis_range=[0 - axis_outer_bound_offset, 2 * np.pi + axis_outer_bound_offset], aspectmode='cube')
    fig['layout']['title'] = rf'$\text{{Plots of the six parameters that make up our representation }}\mathcal{{S}}_{{{sym_cls}}}, \kappa_{{{sym_cls}}}={sym_v} \text{{ ("XYZ" intrinsic rotation order, SO(3) space, angles in radians)}}$'
    #fig_3['layout']['dragmode'] = 'orbit'

    fig.show()
    #fig.write_html(f"{sym_cls}.html", include_mathjax='cdn')


if __name__ == '__main__':
    sym_classes = ['II']#, 'II', 'III', 'IV', 'V']
    tless_steps =  36  # For dots, datapoints, time 2 plus 1 i.e. 50 -> 101
    tless_visu_steps = 3
    so3_steps = 32  # For dots, datapoints, time 2 plus 1 i.e. 50 -> 101
    so3_visu_steps = 4
    colours = 'RGB'
    subspace = 'BOTH'# ['TLESS', 'SO3', 'BOTH']  if  not TLESS, visualizes across all of SO(3) - this can take a long time to load
    #colours = 'grey'

    for sym_cls in sym_classes:
        if subspace == 'TLESS':
            plot_mapping_tless(sym_cls=sym_cls, steps=tless_steps, visu_steps=tless_visu_steps, colours=colours)
            print(f'Plots of the six parameters that make up our representation S_{sym_cls}, with kappa_{sym_cls}, TLESS rotation space')
        elif subspace == 'SO3':
            plot_mapping_whole(sym_cls=sym_cls, steps=so3_steps, visu_steps=so3_visu_steps, colours=colours)
            print(f'Plots of the six parameters that make up our representation S_{sym_cls}, with kappa_{sym_cls}, SO(3) rotation space')
        elif subspace == 'BOTH':
            plot_mapping_tless(sym_cls=sym_cls, steps=tless_steps, visu_steps=tless_visu_steps, colours=colours)
            print(f'Plots of the six parameters that make up our representation S_{sym_cls}, with kappa_{sym_cls}, TLESS rotation space')
            plot_mapping_whole(sym_cls=sym_cls, steps=so3_steps, visu_steps=so3_visu_steps, colours=colours)
            print(f'Plots of the six parameters that make up our representation S_{sym_cls}, with kappa_{sym_cls}, SO(3) rotation space')
        else:
            raise NotImplementedError

