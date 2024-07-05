import math

import numpy as np

from scipy.spatial.transform import Rotation
atol = 0.0000000001


def rotation_matrix(theta, order='XYZ'):
    """
    NOTE: These rotatin matrices correspond to the rotations around Tait-Bryan angles as given by
    https://en.wikipedia.org/wiki/Euler_angles#cite_note-4
    and
    https://ntrs.nasa.gov/api/citations/19770019231/downloads/19770019231.pdf
    BUT (!) - Blender actually does the rotations from O_obj = O_world = I but in reverse order than the one
    given in the object definition. So if you set object-order 'ZYX' in Blender the 'XYZ' matrices from those
    sources are correct.
    Since this leads to even more confusion, instead I have decided to adopt the ZYX matrices as is, but switch
    the usage of cx <-> cz & sx <-> sz
    input
        theta1, theta2, theta3 = rotation angles in rotation order (degrees)
        order = rotation order of x,y,z　e.g. XZY rotation -- 'xzy'
    output
        3x3 rotation matrix (numpy array)
    """

    cx, cy, cz = np.cos(theta)
    sx, sy, sz = np.sin(theta)

    if order == 'XYZ':  # intrinsic order NOT Blender order
        matrix = np.array([[cy * cz, -cy * sz, sy],
                           [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
                           [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy]], dtype=float)
    elif order == 'ZYX':
        matrix = np.array([[cy * cx, cx * sz * sy - cz * sx, cz * cx * sy + sz * sx],
                           [cy * sx, cz * cx + sz * sy * sx, cz * sy * sx - cx * sz],
                           [-sy, cy * sz, cz * cy]], dtype=float)
    else:
        print('weird order')
        raise NotImplementedError

    return matrix


SYMMETRY_CLASSES = {
    # class_name, class_id, n_x, n_y, n_z, matrices that lead to unambiguous rotation (NOT continuous)
    'A': {'sym_v': np.array([1, 1, 1])},
    'B': {'sym_v': np.array([1, 1, 2])},
    'C': {'sym_v': np.array([1, 1, 4])},
    'D': {'sym_v': np.array([1, 1, 10 ** 6])},
    'E': {'sym_v': np.array([1, 2, 1])},
    'SPH': {'sym_v': np.array([10 ** 6, 10 ** 6, 10 ** 6])},
    'CUBE': {'sym_v': np.array([4, 4, 4])},
    'CUBOID_XY': {'sym_v': np.array([2, 2, 4])},
    'CUBOID_XZ': {'sym_v': np.array([2, 4, 2])},
    'CUBOID_YZ': {'sym_v': np.array([4, 2, 2])},
    'CUBOID': {'sym_v': np.array([2, 2, 2])},
    'CON': {'sym_v': np.array([1, 1, 10 ** 6])},
    'CYL': {'sym_v': np.array([2, 2, 10 ** 6])},
    'TOR': {'sym_v': np.array([2, 2, 10 ** 6])},
    'NONE': {'sym_v': np.array([1, 1, 1])},
}


TLESS_SCENES_TO_OBJ_CLASSES = {
    1: [2, 25, 29, 30],
    2: [5, 6, 7],
    3: [5, 8, 11, 12, 18],
    4: [5, 8, 26, 28],
    5: [1, 4, 9, 10, 27],
    6: [6, 7, 11, 12],
    7: [1, 3, 13, 14, 15, 16, 17, 18],
    8: [19, 20, 21, 22, 23, 24],
    9: [1, 2, 3, 4],
    10: [19, 20, 21, 22, 23, 24],
    11: [5, 8, 9, 10],
    12: [2, 3, 7, 9],
    13: [19, 20, 21, 23, 28],
    14: [19, 20, 22, 23, 24],
    15: [25, 26, 27, 28, 29, 30],
    16: [10, 11, 12, 13, 14, 15, 16, 17],
    17: [1, 4, 7, 9],
    18: [1, 4, 7, 9],
    19: [13, 14, 15, 16, 17, 18, 24, 30],
    20: [1, 2, 3, 4]
}

TLESS_SYM_CLASSES = {
    0: {'sym_v': np.array([1, 1, 1])},
    1: {'sym_v': np.array([1, 1, 2])},
    2: {'sym_v': np.array([1, 1, 4])},
    3: {'sym_v': np.array([1, 1, 10 ** 6])},
    4: {'sym_v': np.array([1, 2, 1])}
}

TLESS_SYM_CLASSES_TO_OBJ_CLASSES = {
    0: [21, 22, 18],
    1: [11, 5, 6, 7, 8, 9, 10, 12, 25, 26, 28, 29],
    2: [27],
    3: [2, 17, 1, 3, 4, 13, 14, 15, 16, 24, 30],
    4: [23, 19, 20]
}

TLESS_CLASSES = {
    # class id, corresponding to this image: http://cmp.felk.cvut.cz/t-less/, symmetry vector, object dimensions, average distance
    1: {'sym_cls': 3,
        'sym_v': np.array([1, 1, 10 ** 6]),
        'dims': np.array([0.0349916, 0.0349916, 0.0612]),
        'avg_tz': 645.334,
         'mean': 149.172,
         'stddev': 958.763},
    2: {'sym_cls': 3,
        'sym_v': np.array([1, 1, 10 ** 6]),
        'dims': np.array([0.0432896, 0.0432896, 0.0617022]),
        'avg_tz': 639.084,
         'mean': 206.568,
         'stddev': 1116.225},
    3: {'sym_cls': 3,
        'sym_v': np.array([1, 1, 10 ** 6]),
        'dims': np.array([0.0477627, 0.0477627, 0.0616702]),
        'avg_tz': 632.499,
         'mean': 220.262,
         'stddev': 1148.054},
    4: {'sym_cls': 3,
        'sym_v': np.array([1, 1, 10 ** 6]),
        'dims': np.array([0.0399956, 0.0399956, 0.078]),
        'avg_tz': 643.207,
         'mean': 218.176,
         'stddev': 1136.040},
    5: {'sym_cls': 1,
        'sym_v': np.array([1, 1, 2]),
        'dims': np.array([0.095, 0.0535, 0.059]),
        'avg_tz': 641.347,
        'mean': 626.517,
        'stddev': 1867.221},
    6: {'sym_cls': 1,
        'sym_v': np.array([1, 1, 2]),
        'dims': np.array([0.0894, 0.050, 0.0555]),
        'avg_tz': 642.393,
        'mean': 560.992,
        'stddev': 1778.403},
    7: {'sym_cls': 1,
        'sym_v': np.array([1, 1, 2]),
        'dims': np.array([0.15, 0.0894, 0.0615]),
        'avg_tz': 639.473,
        'mean': 1406.582,
        'stddev': 2572.25},
    8: {'sym_cls': 1,
        'sym_v': np.array([1, 1, 2]),  # approximation to T-LESS anno, i.e. they put weird matrix
        'dims': np.array([0.1860786, 0.1053344, 0.0600146]),
        'avg_tz': 633.168,
         'mean': 1943.756,
         'stddev': 2815.783},
    9: {'sym_cls': 1,
        'sym_v': np.array([1, 1, 2]),
        'dims': np.array([0.12125, 0.0785, 0.062698]),
        'avg_tz': 643.222,
         'mean': 887.492,
         'stddev': 2167.620},
    10: {'sym_cls': 1,
         'sym_v': np.array([1, 1, 2]),  # approximation to T-LESS anno, i.e. they put weird matrix
         'dims': np.array([0.0806598, 0.0420072, 0.0635026]),
         'avg_tz': 632.717,
         'mean': 383.766,
         'stddev': 1491.288},
    11: {'sym_cls': 1,
         'sym_v': np.array([1, 1, 2]),  # approximation to T-LESS anno, i.e. they put weird matrix
         'dims': np.array([0.0662912, 0.048242, 0.0553]),
         'avg_tz': 638.572,
         'mean': 285.683,
         'stddev': 1309.957},
    12: {'sym_cls': 1,
         'sym_v': np.array([1, 1, 2]),  # approximation to T-LESS anno, i.e. they put weird matrix
         'dims': np.array([0.0783312, 0.057627, 0.0566]),
         'avg_tz': 633.812,
         'mean': 414.616,
         'stddev': 1552.916},
    13: {'sym_cls': 3,
         'sym_v': np.array([1, 1, 10 ** 6]),
         'dims': np.array([0.0399942, 0.0399886, 0.046]),
         'avg_tz': 634.204,
         'mean': 187.970,
         'stddev': 1068.867},
    14: {'sym_cls': 3,
         'sym_v': np.array([1, 1, 10 ** 6]),
         'dims': np.array([0.0441346, 0.0441346, 0.0651]),
         'avg_tz': 640.995,
         'mean': 245.870,
         'stddev': 1211.332},
    15: {'sym_cls': 3,
         'sym_v': np.array([1, 1, 10 ** 6]),
         'dims': np.array([0.0445476, 0.0445542, 0.055]),
         'avg_tz': 634.416,
         'mean': 256.095,
         'stddev': 1235.481},
    16: {'sym_cls': 3,
         'sym_v': np.array([1, 1, 10 ** 6]),
         'dims': np.array([0.055676, 0.0552634, 0.047]),
         'avg_tz': 635.138,
         'mean': 315.569,
         'stddev': 1369.446},
    17: {'sym_cls': 3,
         'sym_v': np.array([1, 1, 10 ** 6]),
         'dims': np.array([0.1077742, 0.1077632, 0.0601]),
         'avg_tz': 640.876,
         'mean': 896.311,
         'stddev': 2181.749},
    18: {'sym_cls': 0,
         'sym_v': np.array([1, 1, 1]),
         'dims': np.array([0.098721, 0.0987388, 0.0637738]),
         'avg_tz': 643.299,
         'mean': 766.596,
         'stddev': 2033.973},
    19: {'sym_cls': 4,
         'sym_v': np.array([1, 2, 1]),
         'dims': np.array([0.0655, 0.0765, 0.047]),
         'avg_tz': 641.091,
         'mean': 415.787,
         'stddev': 1561.939},
    20: {'sym_cls': 4,
         'sym_v': np.array([1, 2, 1]),
         'dims': np.array([0.083, 0.0755, 0.047]),
         'avg_tz': 640.773,
         'mean': 483.344,
         'stddev': 1674.486},
    21: {'sym_cls': 0,
         'sym_v': np.array([1, 1, 1]),
         'dims': np.array([0.0769, 0.0789782, 0.043]),
         'avg_tz': 628.187,
         'mean': 428.395,
         'stddev': 1585.788},
    22: {'sym_cls': 0,
         'sym_v': np.array([1, 1, 1]),
         'dims': np.array([0.0769, 0.0789782, 0.043987]),
         'avg_tz': 636.554,
         'mean': 436.676,
         'stddev': 1593.805},
    23: {'sym_cls': 4,
         'sym_v': np.array([1, 2, 1]),
         'dims': np.array([0.1379484, 0.0735, 0.0520652]),
         'avg_tz': 632.685,
         'mean': 690.675,
         'stddev': 1955.052},
    24: {'sym_cls': 3,
         'sym_v': np.array([1, 1, 10 ** 6]),
         'dims': np.array([0.0429954, 0.042997, 0.0806]),
         'avg_tz': 635.097,
         'mean': 301.288,
         'stddev': 1324.964},
    25: {'sym_cls': 1,
         'sym_v': np.array([1, 1, 2]),
         'dims': np.array([0.096, 0.0615, 0.0608012]),
         'avg_tz': 638.533,
         'mean': 642.500,
         'stddev': 1872.010},
    26: {'sym_cls': 1,
         'sym_v': np.array([1, 1, 2]),
         'dims': np.array([0.096, 0.0615, 0.0607854]),
         'avg_tz': 627.153,
         'mean': 642.273,
         'stddev': 1873.378},
    27: {'sym_cls': 2,
         'sym_v': np.array([1, 1, 4]),
         'dims': np.array([0.1085, 0.1085, 0.056]),
         'avg_tz': 636.807,
         'mean': 1199.291,
         'stddev': 2411.210},
    28: {'sym_cls': 1,
         'sym_v': np.array([1, 1, 2]),
         'dims': np.array([0.995, 0.0995, 0.0484]),
         'avg_tz': 640.137,
         'mean': 814.748,
         'stddev': 2087.994},
    29: {'sym_cls': 1,
         'sym_v': np.array([1, 1, 2]),
         'dims': np.array([0.113, 0.078, 0.0568]),
         'avg_tz': 645.068,
         'mean': 953.811,
         'stddev': 2207.878},
    30: {'sym_cls': 3,
         'sym_v': np.array([1, 1, 10 ** 6]),
         'dims': np.array([0.080, 0.080, 0.0506]),
         'avg_tz': 645.443,
         'mean': 543.331,
         'stddev': 1747.099}
}


def clamp_rot(alpha, beta, gamma, sym_v=None):
    if sym_v is not None:
        if sym_v[1] > 1:
            if alpha % (2 * np.pi) > np.pi:
                alpha = (alpha - np.pi) % (2 * np.pi)
                gamma = (np.pi - gamma) % (2 * np.pi)
                beta *= -1
        else:
            alpha = (alpha % (2 * np.pi / sym_v[0])) * ((sym_v[0] % 10 ** 6) / sym_v[0])
            beta = (beta % (2 * np.pi / sym_v[1])) * ((sym_v[1] % 10 ** 6) / sym_v[1])
            gamma = (gamma % (2 * np.pi / sym_v[2])) * ((sym_v[2] % 10 ** 6) / sym_v[2])

        angs = [alpha, beta, gamma]

        for idx, ang in enumerate(angs):
            if np.isclose(2 * np.pi, ang, atol=atol) or np.isclose(0.0, ang, atol=atol):
                ang = 0.0
            angs[idx] = ang

        alpha = angs[0]
        beta = angs[1]
        gamma = angs[2]
    else:
        alpha %= 2 * np.pi
        beta %= 2 * np.pi
        gamma %= 2 * np.pi

    return alpha, beta, gamma


# From regular euler angles construct sym aware representation
# ASSUMES INTRINSIC XYZ ORDER
def sym_aware_rotation(alpha, beta, gamma, sym_class, clamp=False):
    if type(sym_class) is np.ndarray:
        sym_v = sym_class
    else:
        sym_v = SYMMETRY_CLASSES[sym_class][1]

    if clamp:
        alpha, beta, gamma = clamp_rot(alpha, beta, gamma, sym_v)
    c_a = math.cos(alpha)
    c_b = math.cos(beta)
    c_g = math.cos(gamma)

    sa = math.sin(alpha)
    ca = math.cos(alpha)
    sb = math.sin(beta)
    cb = math.cos(beta)
    sg = math.sin(gamma)
    cg = math.cos(gamma)


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

        s_g_ = math.sin(gamma * (sym_v[2] % (10 ** 6)))
        c_g_ = math.cos(gamma * (sym_v[2] % (10 ** 6)))
    elif sym_v[1] > 1 and sym_v[0] == 1 and sym_v[2] == 1:
        s_a_ = math.sin(alpha)
        c_a_ = math.cos(alpha)

        s_b_ = math.sin(beta * (sym_v[1] % (10 ** 6)))
        c_b_ = math.cos(beta * (sym_v[1] % (10 ** 6)))

        s_g_ = math.sin(gamma) * c_b
        c_g_ = math.cos(gamma)
    elif sym_v[0] > 1 and sym_v[1] == 1 and sym_v[2] == 1:
        s_a_ = math.sin(alpha * (sym_v[0] % (10 ** 6)))
        c_a_ = math.cos(alpha * (sym_v[0] % (10 ** 6)))

        s_b_ = math.sin(beta) * c_a
        c_b_ = math.cos(beta)

        s_g_ = math.sin(gamma) * c_a
        c_g_ = math.cos(gamma)
    elif np.any(sym_v == 1):
        raise NotImplementedError
    else:
        s_a_ = math.sin(alpha * (sym_v[0] % (10 ** 6)))
        c_a_ = math.cos(alpha * (sym_v[0] % (10 ** 6)))

        s_b_ = math.sin(beta * (sym_v[1] % (10 ** 6))) * c_a
        c_b_ = math.cos(beta * (sym_v[1] % (10 ** 6)))

        s_g_ = math.sin(gamma * (sym_v[2] % (10 ** 6))) * c_a * c_b
        c_g_ = math.cos(gamma * (sym_v[2] % (10 ** 6)))

    x_vec = np.expand_dims(np.round(np.array([s_a_, c_a_]), 10), axis=1)
    y_vec = np.expand_dims(np.round(np.array([s_b_, c_b_]), 10), axis=1)
    z_vec = np.expand_dims(np.round(np.array([s_g_, c_g_]), 10), axis=1)

    rot_sym_mat = np.concatenate((x_vec, y_vec, z_vec), axis=1)

    return rot_sym_mat

def sym_class_to_sym_v(sym_class):
    if isinstance(sym_class, str):
        sym_v = SYMMETRY_CLASSES[sym_class][1]
    else:
        sym_v = SYMMETRY_CLASSES[
            list(SYMMETRY_CLASSES.keys())[(sym_class.item() if type(sym_class) is not int else sym_class)]][1]

    return sym_v


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


def map_tless_R_to_own_R(R, sym_v, clamp=False):
    rot_sym_mat = map_tless_R_to_rot_sym_mat(R, sym_v, clamp)
    alpha, beta, gamma = inv_sym_aware_rotation(rot_sym_mat, sym_v)
    alpha %= 2 * np.pi
    beta %= 2 * np.pi
    gamma %= 2 * np.pi
    R = Rotation.from_euler('XYZ', [alpha, beta, gamma], degrees=False)
    R_mapped = R.as_matrix()

    return R_mapped


def map_tless_R_to_mod_tless_R(R, sym_v, clamp=False):
    rot_sym_mat = map_tless_R_to_rot_sym_mat(R, sym_v, clamp)
    R = map_rot_sym_mat_to_tless_R(rot_sym_mat, sym_v, clamp)

    return R


def map_tless_R_to_rot_sym_mat(R, sym_v, clamp=False):
    alpha, beta, gamma = map_tless_R_to_eulers(R, sym_v)
    rot_sym_mat = sym_aware_rotation(alpha, beta, gamma, sym_v, clamp=clamp)

    return rot_sym_mat


def map_tless_R_to_eulers(R, sym_v):
    R = Rotation.from_matrix(R)
    alpha, beta, gamma = R.as_euler('XYZ', degrees=True)
    alpha, beta, gamma = R.as_euler('XYZ', degrees=False)

    return alpha, beta, gamma


def map_rot_sym_mat_to_tless_R(rot_sym_mat, sym_v, clamp=False):
    alpha, beta, gamma = inv_sym_aware_rotation(rot_sym_mat, sym_v)

    alpha %= 2 * np.pi
    beta %= 2 * np.pi
    gamma %= 2 * np.pi
    R = rotation_matrix((alpha, beta, gamma))

    return R


if __name__ == '__main__':
    sym_class = np.array([1, 2, 1])
    alpha = np.deg2rad(50)
    beta = np.deg2rad(180)
    gamma = np.deg2rad(135)

    rot_sym_mat = sym_aware_rotation(alpha, beta, gamma, sym_class)
    print(rot_sym_mat)
