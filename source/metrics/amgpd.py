import os
import copy
import yaml
import json
import numpy as np

from scipy.optimize import linear_sum_assignment

from source.utils.utils import easydict_constructor, rotational_error, unpack_csv_gt, unpack_csv_pred


T_LESS_sym_cls_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 23, 24, 25, 26, 27, 28, 29, 30]
tless_path = '/mnt/01_Disk/krieglera/RAL/tless'
itodd_path = '/mnt/01_Disk/krieglera/RAL/itodd'

def get_test_targets():
    with open(tless_path + r'base/test_targets_bop19.json', 'r') as f:
        yaml.add_constructor('tag:yaml.org,2002:python/object/new:easydict.EasyDict', easydict_constructor)
        test_targets_raw = yaml.load(f, Loader=yaml.FullLoader)

    test_targets = []
    for entry in test_targets_raw:
        test_targets.append(f"{str(entry['scene_id'])}-{str(entry['im_id'])}-{str(entry['obj_id'])}")

    return test_targets


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


def calc_amgpd_distance(R_gt, t_gt, R_pred, t_pred, model_points):
    """
    Compute A(M)GPD distance for a single prediction.
    The decision to use mean or max depends on model_points (group structure).
    """
    group_dists = []
    for j, group in enumerate(model_points):
        _, n_p = group.shape
        group *= 1.0

        npt = np.tile(t_pred[:, np.newaxis], (1, n_p)) * 0.0001
        ntt = np.tile(t_gt[:, np.newaxis], (1, n_p)) * 0.0001

        pred_pts = R_pred @ group + npt
        gt_pts = R_gt @ group +ntt

        pred_pts_n = np.tile(pred_pts[np.newaxis, :], (n_p, 1, 1))
        gt_pts_n = np.tile(gt_pts.T[:, :, np.newaxis], (1, 1, n_p))

        diff = pred_pts_n - gt_pts_n  # (num_p, 3, num_p)
        norm = np.linalg.norm(diff, axis=1)  # (num_p, num_p)

        min_dist = np.min(norm, axis=1)  # (num_p,)

        if len(model_points) == 3 and j == 2:
            group_dists.append(np.max(min_dist))
        else:
            group_dists.append(np.mean(min_dist))

    if len(model_points) == 3 and model_points[2].shape[1] > 1:
        return float(np.max(group_dists).item())
    else:
        return float(np.mean(group_dists).item())


def calc_amgpd(gt_rots, gt_trans, gt_clss, pd_rots, pd_trans, pd_clss, model_groups_list, n_cls=31):
    cls_add_dis = [[] for _ in range(0, n_cls)]
    cls_adds_dis = [[] for _ in range(0, n_cls)]
    for k, img_gt_rots in gt_rots.items():
        try:
            img_pd_rots = pd_rots[k]
            img_gt_trans = gt_trans[k]
            img_pd_trans = pd_trans[k]
            img_gt_cls = gt_clss[k]
            img_pd_cls = pd_clss[k]
        except KeyError:
            print(k)
        for gt_rot, gt_t, gt_cls, pd_rot, pd_t, pd_cls in zip(img_gt_rots, img_gt_trans, img_gt_cls, img_pd_rots, img_pd_trans, img_pd_cls):
            cls_add_dis, cls_adds_dis = evaluate_amgpd(cls_add_dis, cls_adds_dis, gt_rot, gt_t, gt_cls, pd_rot, pd_t, pd_cls, model_groups_list, n_cls=31)

    return cls_add_dis, cls_adds_dis


def evaluate_amgpd(cls_add_dis, cls_adds_dis, gt_rot, gt_trans, gt_cls, pd_rot, pd_trans, pd_cls, model_groups_list, n_cls=31):
    """
    Returns cls_add_dis and cls_adds_dis lists just like the original.
    """
    if pd_cls == 0 or pd_cls != gt_cls:
        return cls_add_dis, cls_adds_dis

    model_groups = copy.deepcopy(model_groups_list[pd_cls.item()-1])  # assumes 0-based class indexing

    try:
        dist = calc_amgpd_distance(gt_rot, gt_trans, pd_rot, pd_trans, model_groups)
    except ValueError:
        dist = 9999

    cls_add_dis[pd_cls.item()].append(dist)
    cls_adds_dis[pd_cls.item()].append(dist)
    cls_add_dis[0].append(dist)
    cls_adds_dis[0].append(dist)

    return cls_add_dis, cls_adds_dis


def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10

    return ap


def compute_auc(distances, max_dist=0.05, thr_m=0.01):
    D = np.array(distances)
    D[np.where(D > max_dist)] = np.inf
    D = np.sort(D)
    if D.size == 0:
        return 0.0, 0.0
    n = len(distances)
    acc = np.cumsum(np.ones((1, n)), dtype=np.float32) / n
    aps = VOCap(D, acc)

    add_t_cm = np.where(D < thr_m)[0].size / D.size

    return aps * 100, add_t_cm * 100


def summarize_amgpd(cls_add_dis, cls_adds_dis, sym_cls_ids):
    n_cls = len(cls_add_dis)
    cls_add_s_dis = [[] for _ in range(n_cls)]
    add_auc_lst = []
    adds_auc_lst = []
    add_s_auc_lst = []

    add_2cm_lst = []
    adds_2cm_lst = []
    add_s_2cm_lst = []

    for cls_id in range(1, n_cls):
        if cls_id in sym_cls_ids:
            cls_add_s_dis[cls_id] = cls_adds_dis[cls_id]
        else:
            cls_add_s_dis[cls_id] = cls_add_dis[cls_id]
        cls_add_s_dis[0] += cls_add_s_dis[cls_id]

    for i in range(n_cls):
        add_auc, add_2cm = compute_auc(cls_add_dis[i])
        adds_auc, adds_2cm = compute_auc(cls_adds_dis[i])
        add_s_auc, add_s_2cm = compute_auc(cls_add_s_dis[i])

        add_auc_lst.append(add_auc)
        adds_auc_lst.append(adds_auc)
        add_s_auc_lst.append(add_s_auc)

        add_2cm_lst.append(add_2cm)
        adds_2cm_lst.append(adds_2cm)
        add_s_2cm_lst.append(add_s_2cm)

    results = {
        "A(M)GPD AUC": np.mean(add_auc_lst[1:]),
        "A(M)GPD-S AUC": np.mean(adds_auc_lst[1:]),
        "A(M)GPD(-S) AUC": np.mean(add_s_auc_lst[1:]),
        "<2cm A(M)GPD": np.mean(add_2cm_lst[1:]),
        "<2cm A(M)GPD-S": np.mean(adds_2cm_lst[1:]),
        "<2cm A(M)GPD(-S)": np.mean(add_s_2cm_lst[1:]),
        "Overall A(M)GPD(-S) AUC": add_s_auc_lst[0],
    }

    return results


def load_grouped_primitives(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    all_models = []
    for obj in data:
        groups = []
        for grp in obj['groups']:
            grp_np = np.array(grp).T  # Convert to (3, N)
            groups.append(grp_np.astype(np.float32))
        all_models.append(groups)

    return all_models


def main():
    eval_files_tless_other = \
        [
            'results/T-LESS/others/hodan-iros15_tless-test-primesense.csv',
            'results/T-LESS/others/sundermeyer-ijcv19icp_tless-test.csv',
            'results/T-LESS/others/gdrnpp-pbrreal-rgbd-mmodel_tless-test.csv',
            'results/T-LESS/others/zebraposesat-effnetb4-refineddefaultdetections-2023_tless-test.csv',
            'results/T-LESS/others/modalocclusion-rgbd_tless-test.csv',
            'results/T-LESS/others/gpose2023_tless-test.csv',
            'results/T-LESS/others/sc6d-pbr_tless-test.csv',
            'results/T-LESS/others/drost-cvpr10-3d-only_tless-test.csv',
            'results/T-LESS/others/vidal-sensors18_tless-test.csv',
            'results/T-LESS/others/zte-ppf_tless-test.csv',
            'results/T-LESS/others/leroy-fuseocclu-depth_tless-test.csv'
        ]
    eval_files_tless_ours = \
        [
            'results/T-LESS/ours/6d/canonic/6d-canon-dataset_tless-test.csv',
            'results/T-LESS/ours/6d/canonic/6d-canon-object_tless-test.csv',
            'results/T-LESS/ours/6d/canonic/6d-canon-symmetry_tless-test.csv',
            'results/T-LESS/ours/6d/default/6d-default-dataset_tless-test.csv',
            'results/T-LESS/ours/6d/default/6d-default-object_tless-test.csv',
            'results/T-LESS/ours/6d/default/6d-default-symmetry_tless-test.csv',
            'results/T-LESS/ours/euler/canonic/euler-canon-dataset_tless-test.csv',
            'results/T-LESS/ours/euler/canonic/euler-canon-object_tless-test.csv',
            'results/T-LESS/ours/euler/canonic/euler-canon-symmetry_tless-test.csv',
            'results/T-LESS/ours/euler/default/euler-default-dataset_tless-test.csv',
            'results/T-LESS/ours/euler/default/euler-default-object_tless-test.csv',
            'results/T-LESS/ours/euler/default/euler-default-symmetry_tless-test.csv',
            'results/T-LESS/ours/quaternion/canonic/quat-canon-dataset_tless-test.csv',
            'results/T-LESS/ours/quaternion/canonic/quat-canon-object_tless-test.csv',
            'results/T-LESS/ours/quaternion/canonic/quat-canon-symmetry_tless-test.csv',
            'results/T-LESS/ours/quaternion/default/quat-default-dataset_tless-test.csv',
            'results/T-LESS/ours/quaternion/default/quat-default-object_tless-test.csv',
            'results/T-LESS/ours/quaternion/default/quat-default-symmetry_tless-test.csv',
            'results/T-LESS/ours/rotation-matrix/canonic/rotmat-canon-dataset_tless-test.csv',
            'results/T-LESS/ours/rotation-matrix/canonic/rotmat-canon-object_tless-test.csv',
            'results/T-LESS/ours/rotation-matrix/canonic/rotmat-canon-symmetry_tless-test.csv',
            'results/T-LESS/ours/rotation-matrix/default/rotmat-default-dataset_tless-test.csv',
            'results/T-LESS/ours/rotation-matrix/default/rotmat-default-object_tless-test.csv',
            'results/T-LESS/ours/rotation-matrix/default/rotmat-default-symmetry_tless-test.csv',
            'results/T-LESS/ours/trigonometric/canonic/trig-canon-dataset_tless-test.csv',
            'results/T-LESS/ours/trigonometric/canonic/trig-canon-object_tless-test.csv',
            'results/T-LESS/ours/trigonometric/canonic/trig-canon-symmetry_tless-test.csv',
            'results/T-LESS/ours/trigonometric/default/trig-default-dataset_tless-test.csv',
            'results/T-LESS/ours/trigonometric/default/trig-default-object_tless-test.csv',
            'results/T-LESS/ours/trigonometric/default/trig-default-symmetry_tless-test.csv',
            'results/T-LESS/ours/SARR/sarr-canon-dataset_tless-test.csv',
            'results/T-LESS/ours/SARR/sarr-canon-object_tless-test.csv',
            'results/T-LESS/ours/SARR/sarr-canon-symmetry_tless-test.csv',
            'results/T-LESS/ours/SARR/gtsymcls-sarr-canon-datasetnohm_tless-test.csv',
        ]
    gt_file = os.path.join(os.getcwd(), 'results/T-LESS/gt/tless_gt_bop19_canonic-test.csv')
    eval_file_full_paths = [os.path.join(os.getcwd(), file) for file in eval_files_tless_ours]  # eval_files_tless_other, eval_files_tless_ours

    model_groups_list = load_grouped_primitives(tless_path + r'/ES6D/tless_gp.json')
    for e_file in eval_file_full_paths:
        print('--------------------------------')
        #res_file = e_file.replace(e_file.split('//')[-1], 'amgpd_results.txt')
        name = e_file.split('/')[-1]
        print(name)
        for task in ['SiSo', 'ViVo']:
            gt_rots, gt_trans, gt_cls = unpack_csv_gt(gt_file, task)
            pd_rots, pd_trans, pd_cls = unpack_csv_pred(e_file, gt_rots, task, foreign=True if 'others' in e_file else False)

            gt_rots, gt_trans, gt_cls, pd_rots, pd_trans, pd_cls = get_erot_matches(gt_rots, gt_trans, gt_cls, pd_rots, pd_trans, pd_cls)
            cls_add_dis, cls_adds_dis = calc_amgpd(gt_rots, gt_trans, gt_cls, pd_rots, pd_trans, pd_cls, model_groups_list, n_cls=31)

            sym_cls_ids = T_LESS_sym_cls_ids

            results = summarize_amgpd(cls_add_dis, cls_adds_dis, sym_cls_ids)

            for k, v in results.items():
                string = f"{task}: {k}: {100 * np.round(v/100, decimals=3):.1f}"
                if k == r'Overall A(M)GPD(-S) AUC':
                    print(string)
                #with open(res_file, 'a') as f:
                 #   f.write(string + '/n')

if __name__ == "__main__":
    main()
