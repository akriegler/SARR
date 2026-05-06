import numpy as np

from math import pi as PI
from math import sin, cos, acos

from easydict import EasyDict as edict

atol = 0.0000000001


def easydict_constructor(loader, node):
    fields = loader.construct_mapping(node, deep=False)

    return edict(fields)


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


def rotational_error(gt_rot_mat, pd_rot_mat):
    inv_gt_rot = np.linalg.inv(gt_rot_mat)
    matmul = pd_rot_mat @ inv_gt_rot
    trace_minus_1 = np.trace(np.squeeze(matmul)) - 1
    arccos = acos(max(min(trace_minus_1, 2), -2) / 2.0)
    error = arccos * 180 / PI

    return error


def unpack_csv_gt(file, task):
    header = "scene_id"
    with open(file, "r") as f:
        rotations = {}
        translations ={}
        class_ids = {}
        line_id = 0
        prev_obj_id = -1
        prev_img_id = -1
        for line in f:
            line_id += 1
            elems = line.strip().split(",")
            if len(elems) == 1:
                continue
            scene_id = elems[0]
            img_id = elems[1]
            obj_id = elems[2]
            if len(elems) != 7:
                raise ValueError("A line does not have 7 comma-sep. elements: {}".format(line))
            elif line_id == 1 and header in line:
                continue
            # There isnt any rigorously defined method to choose the instance for siso evaluation. One could
            # choose the first one as is done here, or the one with highest visibility, or with most model confidence,
            # or with least pose error in a sorting step. SiSo task is not of a lot of interest anymore anyhow.
            elif obj_id == prev_obj_id and prev_img_id == img_id  and task == 'siso':
                continue
            prev_obj_id = obj_id
            prev_img_id  = img_id
            R = np.array(list(map(float, elems[4].split())), np.float64).reshape((3, 3))
            t = np.array(list(map(float, elems[5].split())), np.float64)
            id = np.array(list(map(float, elems[2].split())), np.int32)
            try:
                rotations[f'{scene_id}-{img_id}-{obj_id}'].append(R)
                translations[f'{scene_id}-{img_id}-{obj_id}'].append(t)
                class_ids[f'{scene_id}-{img_id}-{obj_id}'].append(id)
            except KeyError:
                rotations[f'{scene_id}-{img_id}-{obj_id}'] = []
                rotations[f'{scene_id}-{img_id}-{obj_id}'].append(R)
                translations[f'{scene_id}-{img_id}-{obj_id}'] = []
                translations[f'{scene_id}-{img_id}-{obj_id}'].append(t)
                class_ids[f'{scene_id}-{img_id}-{obj_id}'] = []
                class_ids[f'{scene_id}-{img_id}-{obj_id}'].append(id)

    return rotations, translations, class_ids


def unpack_csv_pred(file, gt_rotations, task, foreign=False):
    header = "scene_id"
    with open(file, "r") as f:
        rotations = {}
        translations = {}
        class_ids = {}
        line_id = 0
        prev_obj_id = -1
        prev_img_id = -1
        for line in f:
            line_id += 1
            elems = line.split(",")
            if len(elems) == 1:
                continue
            scene_id = elems[0]
            img_id = elems[1]
            obj_id = elems[2]
            if foreign and len(elems) != 7:
                raise ValueError("A line does not have 7 comma-sep. elements: {}".format(line))
            elif line_id == 1 and header in line:
                continue
            # There isnt any rigorously defined method to choose the instance for siso evaluation. One could
            # choose the first one as is done here, or the one with highest visibility, or with most model confidence,
            # or with least pose error in a sorting step. SiSo task is not of a lot of interest anymore anyhow.
            elif obj_id == prev_obj_id and prev_img_id == img_id and task == 'siso':
                continue
            elif line.split(',')[4 if foreign else 7].split(' ')[0] == 'nan':
                continue
            if f'{scene_id}-{img_id}-{obj_id}' in gt_rotations:
                R = np.array(list(map(float, elems[4 if foreign else 7].split())), np.float64)
                t = np.array(list(map(float, elems[5 if foreign else 8].split())), np.float64)
                cls_id = np.array(list(map(float, elems[2].split())), np.int32)
                if R.size > 1:
                    R = R.reshape((3, 3))
                else:
                    R = np.eye(3, dtype=np.float64)
                try:
                    rotations[f'{scene_id}-{img_id}-{obj_id}'].append(R)
                    translations[f'{scene_id}-{img_id}-{obj_id}'].append(t)
                    class_ids[f'{scene_id}-{img_id}-{obj_id}'].append(cls_id)
                except KeyError:
                    rotations[f'{scene_id}-{img_id}-{obj_id}'] = []
                    rotations[f'{scene_id}-{img_id}-{obj_id}'].append(R)
                    translations[f'{scene_id}-{img_id}-{obj_id}'] = []
                    translations[f'{scene_id}-{img_id}-{obj_id}'].append(t)
                    class_ids[f'{scene_id}-{img_id}-{obj_id}'] = []
                    class_ids[f'{scene_id}-{img_id}-{obj_id}'].append(cls_id)
                prev_obj_id = obj_id
                prev_img_id = img_id

    return rotations, translations, class_ids
