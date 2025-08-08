import math

import numpy as np

from scipy.spatial.transform import Rotation

from source.utils.utils import clamp_rot, clamp_rot_adv, rotation_matrix

atol = 0.0000000001


def map_R_to_canonic_R(R, sym_v, clamp=False):
    rot_sym_mat = map_R_to_sarr(R, sym_v, clamp)
    R = map_sarr_to_R(rot_sym_mat, sym_v, clamp)

    return R


def map_R_to_sarr(R, sym_v=None, clamp=False):
    alpha, beta, gamma = map_R_to_euler(R)
    rot_sym_mat = sym_aware_rotation(alpha, beta, gamma, sym_v, clamp=clamp)

    return rot_sym_mat


def map_R_to_euler(R, sym_v=None, clamp=False):
    if sym_v is None:
        sym_v = [1, 1, 1]
    R = Rotation.from_matrix(R)
    alpha, beta, gamma = R.as_euler('XYZ', degrees=False)
    if clamp:
        alpha, beta, gamma = clamp_rot(alpha, beta, gamma, sym_v=sym_v)

    return alpha, beta, gamma


def map_sarr_to_R(rot_sym_mat, sym_v, clamp=False):
    alpha, beta, gamma = inv_sym_aware_rotation(rot_sym_mat, sym_v)

    if clamp:
        alpha, beta, gamma = clamp_rot(alpha, beta, gamma, sym_v)

    R = rotation_matrix((alpha, beta, gamma))

    return R


# From regular euler angles construct sym aware representation
# ASSUMES INTRINSIC XYZ ORDER
def sym_aware_rotation(alpha, beta, gamma, sym_class, clamp=False):
    if sym_class is None:
        sym_v = [1, 1, 1]
    elif type(sym_class) is np.ndarray:
        sym_v = sym_class

    if clamp:
        alpha, beta, gamma = clamp_rot_adv(alpha, beta, gamma, sym_v)

    c_a = math.cos(alpha)
    c_b = math.cos(beta)
    #c_g = math.cos(gamma)

    if max(sym_v) == 1:
        s_a_ = math.sin(alpha)
        c_a_ = math.cos(alpha)

        s_b_ = math.sin(beta)
        c_b_ = math.cos(beta)

        s_g_ = math.sin(gamma)
        c_g_ = math.cos(gamma)
    elif sym_v[2] > 1 and sym_v[0] == 1 and sym_v[1] == 1:
        s_a_ = math.sin(alpha)
        c_a_ = math.cos(alpha)

        s_b_ = math.sin(beta)
        c_b_ = math.cos(beta)

        s_g_ = math.sin(gamma * (sym_v[2] % (10 ** 3)))
        c_g_ = math.cos(gamma * (sym_v[2] % (10 ** 3)))
    elif sym_v[1] > 1 and sym_v[0] == 1 and sym_v[2] == 1:
        s_a_ = math.sin(alpha)
        c_a_ = math.cos(alpha)

        s_b_ = math.sin(beta * (sym_v[1] % (10 ** 3)))
        c_b_ = math.cos(beta * (sym_v[1] % (10 ** 3)))

        s_g_ = math.sin(gamma) * c_b
        c_g_ = math.cos(gamma)
    elif sym_v[0] > 1 and sym_v[1] == 1 and sym_v[2] == 1:
        s_a_ = math.sin(alpha * (sym_v[0] % (10 ** 3)))
        c_a_ = math.cos(alpha * (sym_v[0] % (10 ** 3)))

        s_b_ = math.sin(beta) * c_a
        c_b_ = math.cos(beta)

        s_g_ = math.sin(gamma) * c_a
        c_g_ = math.cos(gamma)
    elif np.any(sym_v == 1):
        raise NotImplementedError
    else:
        s_a_ = math.sin(alpha * (sym_v[0] % (10 ** 3)))
        c_a_ = math.cos(alpha * (sym_v[0] % (10 ** 3)))

        s_b_ = math.sin(beta * (sym_v[1] % (10 ** 3))) * c_a
        c_b_ = math.cos(beta * (sym_v[1] % (10 ** 3)))

        s_g_ = math.sin(gamma * (sym_v[2] % (10 ** 3))) * c_a * c_b
        c_g_ = math.cos(gamma * (sym_v[2] % (10 ** 3)))

    x_vec = np.expand_dims(np.round(np.array([s_a_, c_a_]), 10), axis=1)
    y_vec = np.expand_dims(np.round(np.array([s_b_, c_b_]), 10), axis=1)
    z_vec = np.expand_dims(np.round(np.array([s_g_, c_g_]), 10), axis=1)

    rot_sym_mat = np.concatenate((x_vec, y_vec, z_vec), axis=1)

    return rot_sym_mat


def inv_sym_aware_rotation(rot_sym_mat, sym_class):
    if type(sym_class) is np.ndarray:
        sym_v = sym_class
    else:
        sym_v = sym_class_to_sym_v(sym_class)

    if max(sym_v) == 1:
        if rot_sym_mat[0, 0] < 0.0:
            alpha = 2 * np.pi - math.acos(rot_sym_mat[1, 0])
        else:
            alpha = math.acos(rot_sym_mat[1, 0])

        if rot_sym_mat[0, 1] < 0.0:
            beta = 2 * np.pi - math.acos(rot_sym_mat[1, 1])
        else:
            beta = math.acos(rot_sym_mat[1, 1])

        if rot_sym_mat[0, 2] < 0.0:
            gamma = 2 * np.pi - math.acos(rot_sym_mat[1, 2])
        else:
            gamma = math.acos(rot_sym_mat[1, 2])

    elif sym_v[2] > 1 and sym_v[0] == 1 and sym_v[1] == 1:
        if rot_sym_mat[0, 2] < 0.0:
            gamma = (2 * np.pi - math.acos(rot_sym_mat[1, 2]))
        else:
            gamma = math.acos(rot_sym_mat[1, 2])
        gamma /= sym_v[2]

        if rot_sym_mat[0, 0] < 0.0:
            alpha = 2 * np.pi - math.acos(rot_sym_mat[1, 0])
        else:
            alpha = math.acos(rot_sym_mat[1, 0])

        if rot_sym_mat[0, 1] < 0.0:
            beta = 2 * np.pi - math.acos(rot_sym_mat[1, 1])
        else:
            beta = math.acos(rot_sym_mat[1, 1])

    elif sym_v[1] > 1 and sym_v[0] == 1 and sym_v[2] == 1:
        if rot_sym_mat[0, 1] < 0.0:
            beta = (2 * np.pi / sym_v[1]) - (math.acos(rot_sym_mat[1, 1]) / sym_v[1])
            bf = -1
        else:
            beta = math.acos(rot_sym_mat[1, 1])
            beta /= sym_v[1]
            bf = 1

        if rot_sym_mat[0, 0] < 0.0:
            alpha = 2 * np.pi - math.acos(rot_sym_mat[1, 0])
        else:
            alpha = math.acos(rot_sym_mat[1, 0])

        if rot_sym_mat[0, 2] < 0.0:
            gamma = 2 * np.pi - math.acos(rot_sym_mat[1, 2])
        else:
            gamma = math.acos(rot_sym_mat[1, 2])
        gamma *= bf

    elif sym_v[0] > 1 and sym_v[2] == 1 and sym_v[1] == 1:
        if rot_sym_mat[0, 0] < 0.0:
            alpha = 2 * np.pi - (math.acos(rot_sym_mat[1, 0]) / sym_v[0])
        else:
            alpha = math.acos(rot_sym_mat[1, 0])
            alpha /= sym_v[0]

        if rot_sym_mat[0, 2] / math.cos(alpha) < 0.0:
            gamma = 2 * np.pi - math.acos(rot_sym_mat[1, 2])
        else:
            gamma = math.acos(rot_sym_mat[1, 2])

        if rot_sym_mat[0, 1] / math.cos(alpha) < 0.0:
            beta = 2 * np.pi - math.acos(rot_sym_mat[1, 1])
        else:
            beta = math.acos(rot_sym_mat[1, 1])
    elif np.any(sym_v == 1):
        raise NotImplementedError
    else:
        if rot_sym_mat[0, 0] < 0.0:
            alpha = 2 * np.pi - math.acos(rot_sym_mat[1, 0])
        else:
            alpha = math.acos(rot_sym_mat[1, 0])
        alpha /= sym_v[0]

        if rot_sym_mat[0, 1] / math.cos(alpha) < 0.0:
            beta = 2 * np.pi - math.acos(rot_sym_mat[1, 1])
        else:
            beta = math.acos(rot_sym_mat[1, 1])
        beta /= sym_v[1]

        if rot_sym_mat[0, 2] / math.cos(beta) / math.cos(alpha) < 0.0:
            gamma = 2 * np.pi - (math.acos(rot_sym_mat[1, 2]) / sym_v[2])
        else:
            gamma = math.acos(rot_sym_mat[1, 2])

        gamma /= sym_v[2]

    return alpha, beta, gamma
