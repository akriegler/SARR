import math
import numpy as np

from math import pi as PI
from math import sin, cos, acos
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import normalize

atol = 0.0000000001


def rotation_matrix(alpha, beta, gamma, order='XYZ'):
    """
    NOTE: These rotatin matrices correspond to the rotations around Tait-Bryan angles as given by
    https://en.wikipedia.org/wiki/Euler_angles#cite_note-4
    and
    https://ntrs.nasa.gov/api/citations/19770019231/downloads/19770019231.pdf
    input
        theta1, theta2, theta3 = rotation angles in rotation order (degrees)
        order = rotation order of x,y,z　e.g. XZY rotation -- 'xzy'
    output
        3x3 rotation matrix (numpy array)
    """
    sx = sin(alpha)
    cx = cos(alpha)
    sy = sin(beta)
    cy = cos(beta)
    sz = sin(gamma)
    cz = cos(gamma)


    if order == 'XYZ':
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


def clamp_rot(alpha, beta, gamma, kappa=None):
    alpha = mod(alpha, (2 * PI / kappa[0]))
    beta = mod(beta, (2 * PI / kappa[1]))
    gamma = mod(gamma, (2 * PI / kappa[2]))

    return alpha, beta, gamma


def mod(a, b):
    return ((a % b) + b) % b


def map_Euler_to_R_canon(alpha, beta, gamma, kappa, clamp=False):
    from source.SARR import sym_aware_rotation
    sarr = sym_aware_rotation(alpha, beta, gamma, kappa, clamp=clamp)
    R_canon = map_sarr_to_R_canon(sarr, kappa, clamp)

    return R_canon


def map_R_to_euler(R, kappa=None, clamp=False):
    if kappa is None:
        kappa = [1, 1, 1]
    R = Rotation.from_matrix(R)
    alpha, beta, gamma = R.as_euler('XYZ', degrees=False)
    if clamp:
        alpha, beta, gamma = clamp_rot(alpha, beta, gamma, kappa=kappa)

    return alpha, beta, gamma


def map_R_to_R_canon(R, kappa, clamp=False):
    alpha, beta, gamma = map_R_to_euler(R)
    R_canon = map_Euler_to_R_canon(alpha, beta, gamma, kappa, clamp=clamp)

    return R_canon


def map_6d_to_R(SixD):
    x = SixD[:, 0]
    y = SixD[:, 1]

    R1 = normalize(x.reshape(1, -1), axis=1).ravel()
    c = np.cross(R1, y)
    R3 = normalize(c.reshape(1, -1), axis=1).ravel()
    R2 = np.cross(R3, R1)

    R = np.zeros((3, 3), dtype=np.float64)
    R[:, 0] = R1
    R[:, 1] = R2
    R[:, 2] = R3

    return R


def map_SixD_to_R_canon(SixD, kappa, clamp=False):
    R = map_6d_to_R(SixD)
    alpha, beta, gamma = map_R_to_euler(R)
    R_canon = map_Euler_to_R_canon(alpha, beta, gamma, kappa, clamp=clamp)

    return R_canon


def map_trig_to_euler(trig):
    if trig[0, 0] < 0.0:
        alpha = 2 * np.pi - math.acos(trig[1, 0])
    else:
        alpha = math.acos(trig[1, 0])

    if trig[0, 1] < 0.0:
        beta = 2 * np.pi - math.acos(trig[1, 1])
    else:
        beta = math.acos(trig[1, 1])

    if trig[0, 2] < 0.0:
        gamma = 2 * np.pi - math.acos(trig[1, 2])
    else:
        gamma = math.acos(trig[1, 2])

    return alpha, beta, gamma


def map_trig_to_R_canon(trig, kappa, clamp=False):
    alpha, beta, gamma = map_trig_to_euler(trig)
    R_canon = map_Euler_to_R_canon(alpha, beta, gamma, kappa, clamp=clamp)

    return R_canon


def map_quat_to_R(quat):
    try:
        R = Rotation.from_quat(quat[0], scalar_first=False)
        R = R.as_matrix()
    except ValueError:
        R = np.eye(3, dtype=np.float64)

    return R


def map_quat_to_R_canon(quat, kappa, clamp=False):
    R = map_quat_to_R(quat)
    alpha, beta, gamma = map_R_to_euler(R)
    R_canon = map_Euler_to_R_canon(alpha, beta, gamma, kappa, clamp=clamp)

    return R_canon


def map_sarr_to_R_canon(sarr, kappa, clamp=False):
    from source.SARR import inv_sym_aware_rotation
    alpha, beta, gamma = inv_sym_aware_rotation(sarr, kappa)

    if clamp:
        alpha, beta, gamma = clamp_rot(alpha, beta, gamma, kappa)

    R = rotation_matrix(alpha, beta, gamma)

    return R
