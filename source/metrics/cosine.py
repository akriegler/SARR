import os
import yaml
import numpy as np

from scipy.optimize import linear_sum_assignment

from source.utils.utils import easydict_constructor, rotational_error, unpack_csv_gt, unpack_csv_pred

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


def get_erot_matches(gt_rots, pd_rots):
    for k, img_gt_rots in gt_rots.items():
        try:
            img_pd_rots = pd_rots[k]
        except KeyError:
            img_pd_rots = []
        img_cost_mat = np.zeros((len(img_gt_rots), len(img_pd_rots)), dtype=np.float64)
        for i, gt_rot in enumerate(img_gt_rots):
            for j, pd_rot in enumerate(img_pd_rots):
                img_cost_mat[i, j] = rotational_error(gt_rot, pd_rot)
        row_ind, col_ind = linear_sum_assignment(img_cost_mat)
        new_entry = []
        col_ind = list(col_ind)
        for gt_index in range(len(img_gt_rots)):
            if gt_index not in list(row_ind):
                new_entry.append(None)
            else:
                try:
                    new_entry.append(img_pd_rots[col_ind.pop(0)])
                except IndexError:
                    new_entry.append(None)
        pd_rots[k] = new_entry

    return gt_rots, pd_rots


def calc_erot_error(gt_rots, pd_rots):
    errors = []
    for k, img_gt_rots in gt_rots.items():
        try:
            img_pd_rots = pd_rots[k]
        except KeyError:
            print(k)
        for gt_rot, pd_rot in zip(img_gt_rots, img_pd_rots):
            if pd_rot is None:
                errors.append(180.0)
            else:
                errors.append(rotational_error(gt_rot, pd_rot))

    return errors


def calc_erot_recall(e_rot, e_th):
    true = 0
    for e in e_rot:
        if e < e_th:
            true += 1

    recall = true / len(e_rot)

    return recall


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
    gt_file_tless = os.path.join(os.getcwd(), 'results/T-LESS/gt/tless_gt_bop19_canonic-test.csv')

    eval_files_itodd_other = \
        [
            'results/ITODD/others/sc6d-pbr_itodd-test.csv'
        ]
    eval_files_itodd_ours = \
        [
            'results/ITODD/ours/6d/canonic/6d-canon-dataset_itodd-test.csv',
            'results/ITODD/ours/6d/canonic/6d-canon-object_itodd-test.csv',
            #'results/ITODD/ours/6d/canonic/6d-canon-symmetry_itodd-test.csv',
          #  'results/ITODD/ours/6d/default/6d-default-dataset_itodd-test.csv',
           # 'results/ITODD/ours/6d/default/6d-default-object_itodd-test.csv',
           # 'results/ITODD/ours/6d/default/6d-default-symmetry_itodd-test.csv',
           # 'results/ITODD/ours/euler/canonic/euler-canon-dataset_itodd-test.csv',
           # 'results/ITODD/ours/euler/canonic/euler-canon-object_itodd-test.csv',
           # 'results/ITODD/ours/euler/canonic/euler-canon-symmetry_itodd-test.csv',
           # 'results/ITODD/ours/euler/default/euler-default-dataset_itodd-test.csv',
           # 'results/ITODD/ours/euler/default/euler-default-object_itodd-test.csv',
           # 'results/ITODD/ours/euler/default/euler-default-symmetry_itodd-test.csv',
           # 'results/ITODD/ours/quaternion/canonic/quat-canon-dataset_itodd-test.csv',
           # 'results/ITODD/ours/quaternion/canonic/quat-canon-object_itodd-test.csv',
           # 'results/ITODD/ours/quaternion/canonic/quat-canon-symmetry_itodd-test.csv',
           # 'results/ITODD/ours/quaternion/default/quat-default-dataset_itodd-test.csv',
           # 'results/ITODD/ours/quaternion/default/quat-default-object_itodd-test.csv',
           # 'results/ITODD/ours/quaternion/default/quat-default-symmetry_itodd-test.csv',
            'results/ITODD/ours/rotation-matrix/canonic/rotmat-canon-dataset_itodd-test.csv',
            'results/ITODD/ours/rotation-matrix/canonic/rotmat-canon-object_itodd-test.csv',
            'results/ITODD/ours/rotation-matrix/canonic/rotmat-canon-symmetry_itodd-test.csv',
            #'results/ITODD/ours/rotation-matrix/default/rotmat-default-dataset_itodd-test.csv',
           # 'results/ITODD/ours/rotation-matrix/default/rotmat-default-object_itodd-test.csv',
           # 'results/ITODD/ours/rotation-matrix/default/rotmat-default-symmetry_itodd-test.csv',
            #'results/ITODD/ours/trigonometric/canonic/trig-canon-dataset_itodd-test.csv',
            #'results/ITODD/ours/trigonometric/canonic/trig-canon-object_itodd-test.csv',
            #'results/ITODD/ours/trigonometric/canonic/trig-canon-symmetry_itodd-test.csv',
            #'results/ITODD/ours/trigonometric/default/trig-default-dataset_itodd-test.csv',
            #'results/ITODD/ours/trigonometric/default/trig-default-object_itodd-test.csv',
            #'results/ITODD/ours/trigonometric/default/trig-default-symmetry_itodd-test.csv',
            #'results/ITODD/ours/SARR/sarr-canon-dataset_itodd-test.csv',
            'results/ITODD/ours/SARR/sarr-canon-object_itodd-test.csv',
            'results/ITODD/ours/SARR/sarr-canon-symmetry_itodd-test.csv',
            'results/ITODD/ours/SARR/gtsymcls-sarr-canon-datasetnohm_itodd-test.csv'
        ]
    gt_file_itodd = os.path.join(os.getcwd(), 'results/ITODD/gt/itodd_gt_bop19_canonic-test.csv')

    gt_file = gt_file_itodd  #gt_file_tless
    eval_file_full_paths = [os.path.join(os.getcwd(), file) for file in eval_files_itodd_ours]    #  eval_files_tless_other, eval_files_tless_ours, eval_files_itodd_other, eval_files_itodd_ours

    for e_file in eval_file_full_paths:
        print('--------------------------------')
        #res_file = e_file.replace(e_file.split('/')[-1], 'e_rot_results.txt')
        name = e_file.split('/')[-1]
        print(name)
        for task in ['SiSo', 'ViVo']:
            gt_rotations, _, _ = unpack_csv_gt(gt_file, task)
            pd_rotations, _, _ = unpack_csv_pred(e_file, gt_rotations, task, foreign=True if 'others' in e_file else False)

            gt_rotations, pd_rotations = get_erot_matches(gt_rotations, pd_rotations)
            e_rot = calc_erot_error(gt_rotations, pd_rotations)

            #for e in e_rot:
               # print(e.item())
            #print(np.mean(e_rot, axis=0))

            recalls = []
            for e_th in [2, 5, 10, 15, 25, 40]:
                recalls.append(calc_erot_recall(e_rot, e_th))

            string = f"{task}: AR: {100 * np.round(np.array(recalls).mean(), decimals=3):.1f}"
            print(string)

            #with open(res_file, 'a') as f:
            #    f.write(string)


if __name__ == "__main__":
    main()
