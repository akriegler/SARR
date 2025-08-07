import math

import numpy as np

from easydict import EasyDict as edict


def easydict_constructor(loader, node):
    fields = loader.construct_mapping(node, deep=False)

    return edict(fields)


def clamp_rot(alpha, beta, gamma, sym_v=None):
    alpha = alpha % (2 * np.pi / sym_v[0])
    beta = beta % (2 * np.pi / sym_v[1])
    gamma = gamma % (2 * np.pi / sym_v[2])

    return alpha, beta, gamma


def clamp_rot_adv(alpha, beta, gamma, sym_v=None):
    if sym_v[1] > 1:
        if alpha % (2 * np.pi) > np.pi:
            alpha = (alpha - np.pi) % (2 * np.pi)
            gamma = (np.pi - gamma) % (2 * np.pi)
            beta *= -1
    else:
        alpha = (alpha % (2 * np.pi / sym_v[0])) * ((sym_v[0] % 10 ** 3) / sym_v[0])
        beta = (beta % (2 * np.pi / sym_v[1])) * ((sym_v[1] % 10 ** 3) / sym_v[1])
        gamma = (gamma % (2 * np.pi / sym_v[2])) * ((sym_v[2] % 10 ** 3) / sym_v[2])

    angs = [alpha, beta, gamma]

    for idx, ang in enumerate(angs):
        if np.isclose(2 * np.pi, ang, atol=atol) or np.isclose(0.0, ang, atol=atol):
            ang = 0.0
        angs[idx] = ang

    alpha = angs[0]
    beta = angs[1]
    gamma = angs[2]

    return alpha, beta, gamma


# Modified from https://github.com/THU-DA-6D-Pose-Group/GDR-Net/blob/main/core/utils/utils.py#L97
def egocentric_to_allocentric(R_ego, t, cam_ray=(0, 0, 1.0)):
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = np.asarray(cam_ray)
    obj_ray = t.copy() / np.linalg.norm(t)
    angle = math.acos(cam_ray.dot(obj_ray))

    # Rotate back by that amount
    if angle > 0:
        rot_mat = axangle2mat(axis=np.cross(cam_ray, obj_ray), angle=-angle)
        allo_rot = np.dot(rot_mat, R_ego)
    else:
        allo_rot = R_ego.copy()

    return allo_rot


# Source: https://github.com/matthew-brett/transforms3d/blob/main/transforms3d/axangles.py
def axangle2mat(axis, angle, is_normalized=False):
    ''' Rotation matrix for rotation angle `angle` around `axis`

    Parameters
    ----------
    axis : 3 element sequence
       vector specifying axis for rotation.
    angle : scalar
       angle of rotation in radians.
    is_normalized : bool, optional
       True if `axis` is already normalized (has norm of 1).  Default False.

    Returns
    -------
    mat : array shape (3,3)
       rotation matrix for specified rotation

    Notes
    -----
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    '''
    x, y, z = axis
    if not is_normalized:
        n = math.sqrt(x*x + y*y + z*z)
        x = x/n
        y = y/n
        z = z/n

    c = math.cos(angle); s = math.sin(angle); C = 1-c
    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC

    arr =  np.array([
        [ x*xC+c,   xyC-zs,   zxC+ys ],
        [ xyC+zs,   y*yC+c,   yzC-xs ],
        [ zxC-ys,   yzC+xs,   z*zC+c ] ], dtype=np.float64)

    return arr


def rotational_error(gt_rot_mat, pd_rot_mat):
    inv_gt_rot = np.linalg.inv(gt_rot_mat)
    matmul = pd_rot_mat @ inv_gt_rot
    trace_minus_1 = np.trace(np.squeeze(matmul)) - 1
    arccos = np.arccos(max(min(trace_minus_1, 2), -2) / 2.0)
    error = arccos * 180 / np.pi

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
            elif obj_id == prev_obj_id and prev_img_id == img_id  and task == 'SiSo':
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
            elif obj_id == prev_obj_id and prev_img_id == img_id and task == 'SiSo':
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


def get_erot_matches(gt_rots, gt_trans, gt_cls, pd_rots, pd_trans, pd_cls):
    for (k, img_gt_rots), (_, img_gt_trans) in zip(gt_rots.items(), gt_trans.items()):
        try:
            img_pd_rots = pd_rots[k]
            img_pd_cls = pd_cls[k]
        except KeyError:
            img_pd_rots = []
            img_pd_cls = []
        img_cost_mat = np.zeros((len(img_gt_rots), len(img_pd_rots)), dtype=np.float64)
        for i, gt_rot in enumerate(img_gt_rots):
            for j, pd_rot in enumerate(img_pd_rots):
                img_cost_mat[i, j] = rotational_error(gt_rot, pd_rot)
        row_ind, col_ind = linear_sum_assignment(img_cost_mat)
        new_entry_rot = []
        new_entry_trans = []
        new_entry_cls = []
        col_ind = list(col_ind)
        row_ind = list(row_ind)
        for gt_index in range(len(img_gt_rots)):
            if gt_index not in row_ind:
                new_entry_rot.append(None)
                new_entry_trans.append(None)
                new_entry_cls.append(None)
            else:
                try:
                    ind = col_ind.pop(0)
                    new_entry_rot.append(img_pd_rots[ind])
                    new_entry_trans.append(img_gt_trans[gt_index])
                    new_entry_cls.append(img_pd_cls[ind])
                except IndexError:
                    new_entry_rot.append(None)
                    new_entry_trans.append(None)
                    new_entry_cls.append(None)

        pd_rots[k] = new_entry_rot
        pd_trans[k] = new_entry_trans
        pd_cls[k] = new_entry_cls

    return gt_rots, gt_trans, gt_cls, pd_rots, pd_trans, pd_cls
