import math
import numpy as np
from scipy.spatial.transform import Rotation

from source.utils.utils import clamp_rot, clamp_rot_adv, rotation_matrix

atol = 0.0000000001


def map_R_to_canonic_R(R, sym_v, clamp=False):
    sarr = map_R_to_sarr(R, sym_v, clamp)
    R = map_sarr_to_R(sarr, sym_v, clamp)

    return R


def map_R_to_sarr(R, sym_v=None, clamp=False):
    alpha, beta, gamma = map_R_to_euler(R)
    sarr = sym_aware_rotation(alpha, beta, gamma, sym_v, clamp=clamp)

    return sarr


def map_R_to_euler(R, sym_v=None, clamp=False):
    if sym_v is None:
        sym_v = [1, 1, 1]
    R = Rotation.from_matrix(R)
    alpha, beta, gamma = R.as_euler('XYZ', degrees=False)
    if clamp:
        alpha, beta, gamma = clamp_rot(alpha, beta, gamma, sym_v=sym_v)

    return alpha, beta, gamma


def map_sarr_to_R(sarr, sym_v, clamp=False):
    alpha, beta, gamma = inv_sym_aware_rotation(sarr, sym_v)

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
    #c_g = math.cos(gamma)  # For other symmetry classes beyon these necessary

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

    sarr = np.concatenate((x_vec, y_vec, z_vec), axis=1)

    return sarr


def inv_sym_aware_rotation(sarr, sym_class):
    if type(sym_class) is np.ndarray:
        sym_v = sym_class
    else:
        sym_v = np.asarray(sym_class)

    if max(sym_v) == 1:
        if sarr[0, 0] < 0.0:
            alpha = 2 * np.pi - math.acos(sarr[1, 0])
        else:
            alpha = math.acos(sarr[1, 0])

        if sarr[0, 1] < 0.0:
            beta = 2 * np.pi - math.acos(sarr[1, 1])
        else:
            beta = math.acos(sarr[1, 1])

        if sarr[0, 2] < 0.0:
            gamma = 2 * np.pi - math.acos(sarr[1, 2])
        else:
            gamma = math.acos(sarr[1, 2])

    elif sym_v[2] > 1 and sym_v[0] == 1 and sym_v[1] == 1:
        if sarr[0, 2] < 0.0:
            gamma = (2 * np.pi - math.acos(sarr[1, 2]))
        else:
            gamma = math.acos(sarr[1, 2])
        gamma /= sym_v[2]

        if sarr[0, 0] < 0.0:
            alpha = 2 * np.pi - math.acos(sarr[1, 0])
        else:
            alpha = math.acos(sarr[1, 0])

        if sarr[0, 1] < 0.0:
            beta = 2 * np.pi - math.acos(sarr[1, 1])
        else:
            beta = math.acos(sarr[1, 1])

    elif sym_v[1] > 1 and sym_v[0] == 1 and sym_v[2] == 1:
        if sarr[0, 1] < 0.0:
            beta = (2 * np.pi / sym_v[1]) - (math.acos(sarr[1, 1]) / sym_v[1])
            bf = -1
        else:
            beta = math.acos(sarr[1, 1])
            beta /= sym_v[1]
            bf = 1

        if sarr[0, 0] < 0.0:
            alpha = 2 * np.pi - math.acos(sarr[1, 0])
        else:
            alpha = math.acos(sarr[1, 0])

        if sarr[0, 2] < 0.0:
            gamma = 2 * np.pi - math.acos(sarr[1, 2])
        else:
            gamma = math.acos(sarr[1, 2])
        gamma *= bf

    elif sym_v[0] > 1 and sym_v[2] == 1 and sym_v[1] == 1:
        if sarr[0, 0] < 0.0:
            alpha = 2 * np.pi - (math.acos(sarr[1, 0]) / sym_v[0])
        else:
            alpha = math.acos(sarr[1, 0])
            alpha /= sym_v[0]

        if sarr[0, 2] / math.cos(alpha) < 0.0:
            gamma = 2 * np.pi - math.acos(sarr[1, 2])
        else:
            gamma = math.acos(sarr[1, 2])

        if sarr[0, 1] / math.cos(alpha) < 0.0:
            beta = 2 * np.pi - math.acos(sarr[1, 1])
        else:
            beta = math.acos(sarr[1, 1])
    elif np.any(sym_v == 1):
        raise NotImplementedError
    else:
        if sarr[0, 0] < 0.0:
            alpha = 2 * np.pi - math.acos(sarr[1, 0])
        else:
            alpha = math.acos(sarr[1, 0])
        alpha /= sym_v[0]

        if sarr[0, 1] / math.cos(alpha) < 0.0:
            beta = 2 * np.pi - math.acos(sarr[1, 1])
        else:
            beta = math.acos(sarr[1, 1])
        beta /= sym_v[1]

        if sarr[0, 2] / math.cos(beta) / math.cos(alpha) < 0.0:
            gamma = 2 * np.pi - (math.acos(sarr[1, 2]) / sym_v[2])
        else:
            gamma = math.acos(sarr[1, 2])

        gamma /= sym_v[2]

    return alpha, beta, gamma
