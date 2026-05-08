import numpy as np

from math import pi as PI
from math import sin, cos, acos
from scipy.spatial.transform import Rotation

from source.utils.rot_utils import clamp_rot, rotation_matrix, mod

atol = 0.0000000001


def map_R_to_canonic_R(R, kappa, clamp=False):
    sarr = map_R_to_sarr(R, kappa, clamp)
    R = map_sarr_to_R(sarr, kappa, clamp)

    return R


def map_R_to_sarr(R, kappa=None, clamp=False):
    alpha, beta, gamma = map_R_to_euler(R)
    sarr = sym_aware_rotation(alpha, beta, gamma, kappa, clamp=clamp)

    return sarr


def map_R_to_euler(R, kappa=None, clamp=False):
    if kappa is None:
        kappa = [1, 1, 1]
    R = Rotation.from_matrix(R)
    alpha, beta, gamma = R.as_euler('XYZ', degrees=False)
    if clamp:
        alpha, beta, gamma = clamp_rot(alpha, beta, gamma, kappa=kappa)

    return alpha, beta, gamma


def map_sarr_to_R(sarr, kappa, clamp=False):
    alpha, beta, gamma = inv_sym_aware_rotation(sarr, kappa)

    if clamp:
        alpha, beta, gamma = clamp_rot(alpha, beta, gamma, kappa)

    R = rotation_matrix(alpha, beta, gamma)

    return R


def clamp_rot_adv(alpha, beta, gamma, kappa=None):
    if kappa[0] == kappa[1] == kappa[2] == 2:
        if alpha > mod(alpha, PI):
            alpha = mod(alpha, PI)
            beta = mod((2 * PI - beta), PI)
            gamma = mod((2 * PI - gamma), PI)
        elif beta > mod(beta, PI):
            alpha = mod(alpha, PI)
            beta = mod(beta, PI)
            gamma = mod((2 * PI - gamma), PI)
        else:
            alpha = mod(alpha, PI)
            beta = mod(beta, PI)
            gamma = mod(gamma, PI)
    else:
        alpha = mod(alpha, (2 * PI / kappa[0])) * (mod(kappa[0], 10 ** 3) / kappa[0])
        beta = mod(beta, (2 * PI / kappa[1])) * (mod(kappa[1], 10 ** 3) / kappa[1])
        gamma = mod(gamma, (2 * PI / kappa[2])) * (mod(kappa[2], 10 ** 3) / kappa[2])

    alpha = 0.0 if np.isclose(2 * PI, alpha, atol=atol) or np.isclose(0.0, alpha, atol=atol) else alpha
    beta = 0.0 if np.isclose(2 * PI, beta, atol=atol) or np.isclose(0.0, beta, atol=atol) else beta
    gamma = 0.0 if np.isclose(2 * PI, gamma, atol=atol) or np.isclose(0.0, gamma, atol=atol) else gamma

    return alpha, beta, gamma

# From regular euler angles construct sym aware representation
# ASSUMES INTRINSIC XYZ ORDER
def sym_aware_rotation(alpha, beta, gamma, sym_class, clamp=False):
    if sym_class is None:
        kappa = [1, 1, 1]
    elif type(sym_class) is np.ndarray:
        kappa = sym_class
    else:
        kappa = [1, 1, 1]
    
    if clamp:
        alpha, beta, gamma = clamp_rot_adv(alpha, beta, gamma, kappa)

    c_a = cos(alpha)
    c_b = cos(beta)

    if max(kappa) == 1:
        s_a_ = sin(alpha)
        c_a_ = cos(alpha)

        s_b_ = sin(beta)
        c_b_ = cos(beta)

        s_g_ = sin(gamma)
        c_g_ = cos(gamma)
    elif kappa[2] > 1 and kappa[0] == 1 and kappa[1] == 1:
        s_a_ = sin(alpha)
        c_a_ = cos(alpha)

        s_b_ = sin(beta)
        c_b_ = cos(beta)

        s_g_ = sin(gamma * mod(kappa[2], 10 ** 3))
        c_g_ = cos(gamma * mod(kappa[2], 10 ** 3))
    elif kappa[1] > 1 and kappa[0] == 1 and kappa[2] == 1:
        s_a_ = sin(alpha)
        c_a_ = cos(alpha)

        s_b_ = sin(beta * mod(kappa[1], 10 ** 3))
        c_b_ = cos(beta * mod(kappa[1], 10 ** 3))

        s_g_ = sin(gamma) * c_b
        c_g_ = cos(gamma)
    else:
        s_a_ = sin(alpha * mod(kappa[0], 10 ** 3))
        c_a_ = cos(alpha * mod(kappa[0], 10 ** 3))

        s_b_ = sin(beta * mod(kappa[1], 10 ** 3)) * c_a
        c_b_ = cos(beta * mod(kappa[1], 10 ** 3))

        s_g_ = sin(gamma * mod(kappa[2], 10 ** 3)) * c_a * c_b
        c_g_ = cos(gamma * mod(kappa[2], 10 ** 3))

    x_vec = np.expand_dims(np.round(np.array([s_a_, c_a_]), 10), axis=1)
    y_vec = np.expand_dims(np.round(np.array([s_b_, c_b_]), 10), axis=1)
    z_vec = np.expand_dims(np.round(np.array([s_g_, c_g_]), 10), axis=1)

    sarr = np.concatenate((x_vec, y_vec, z_vec), axis=1)

    return sarr


def inv_sym_aware_rotation(sarr, sym_class):
    if type(sym_class) is np.ndarray:
        kappa = sym_class
    else:
        kappa = np.asarray(sym_class)

    if max(kappa) == 1:
        if sarr[0, 0] < 0.0:
            alpha = 2 * PI - acos(sarr[1, 0])
        else:
            alpha = acos(sarr[1, 0])

        if sarr[0, 1] < 0.0:
            beta = 2 * PI - acos(sarr[1, 1])
        else:
            beta = acos(sarr[1, 1])

        if sarr[0, 2] < 0.0:
            gamma = 2 * PI - acos(sarr[1, 2])
        else:
            gamma = acos(sarr[1, 2])

    elif kappa[2] > 1 and kappa[0] == 1 and kappa[1] == 1:
        if sarr[0, 0] < 0.0:
            alpha = 2 * PI - acos(sarr[1, 0])
        else:
            alpha = acos(sarr[1, 0])

        if sarr[0, 1] < 0.0:
            beta = 2 * PI - acos(sarr[1, 1])
        else:
            beta = acos(sarr[1, 1])
        
        if sarr[0, 2] < 0.0:
            gamma = (2 * PI - acos(sarr[1, 2]))
        else:
            gamma = acos(sarr[1, 2])
        gamma /= kappa[2]

    elif kappa[1] > 1 and kappa[0] == 1 and kappa[2] == 1:
        if sarr[0, 0] < 0.0:
            alpha = 2 * PI - acos(sarr[1, 0])
        else:
            alpha = acos(sarr[1, 0])

        if sarr[0, 1] < 0.0:
            beta = 2 * PI - acos(sarr[1, 1])
        else:
            beta = acos(sarr[1, 1])
        beta /= kappa[1]

        if sarr[0, 2] < 0.0:
            gamma = (2 * PI - acos(sarr[1, 2]))
        else:
            gamma = acos(sarr[1, 2])
    else:
        if sarr[0, 0] < 0.0:
            alpha = 2 * PI - acos(sarr[1, 0])
        else:
            alpha = acos(sarr[1, 0])
        alpha /= kappa[0]

        if sarr[0, 1] / cos(alpha) < 0.0:
            beta = 2 * PI - acos(sarr[1, 1])
        else:
            beta = acos(sarr[1, 1])
        beta /= kappa[1]

        if sarr[0, 2] / cos(beta) / cos(alpha) < 0.0:
            gamma = 2 * PI - (acos(sarr[1, 2]) / kappa[2])
        else:
            gamma = acos(sarr[1, 2])
        gamma /= kappa[2]

    return alpha, beta, gamma
