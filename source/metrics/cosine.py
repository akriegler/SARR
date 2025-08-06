import yaml
import numpy as np

from scipy.optimize import linear_sum_assignment

from source.utils.utils import easydict_constructor
from source.utils.utils import rotational_error

tless_path = 'NULL'
itodd_path = 'NULL'


def get_test_targets():
    with open(r'F:\IJCV\tless\base\test_targets_bop19.json', 'r') as f:
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
    eval_files = \
        [
        # r'F:\IJCV\tless\results\drost\drost-cvpr10-3d-only_tless-test.csv',
        #r'F:\IJCV\itodd\results\sc6d\sc6d-pbr_itodd-test.csv',
        #
        ]

    #gt_file = tless_path + r'\results\gt\gt_tless_bop19_canonic-test.csv'
    gt_file = itodd_path + r'\results\gt\gt_itodd_bop19_ego-canonic-test.csv'
    for e_file in eval_files:
        print('--------------------------------')
        res_file = e_file.replace(e_file.split('\\')[-1], 'e_rot_results.txt')
        for task in ['siso', 'vivo']:
            gt_rotations = unpack_csv_gt_rotation(gt_file, task)
            pd_rotations = unpack_csv_pred_rotation(e_file, gt_rotations, task, foreign=False if 'kriegler' in e_file else True)

            gt_rotations, pd_rotations = get_erot_matches(gt_rotations, pd_rotations)
            e_rot = calc_erot_error(gt_rotations, pd_rotations)
            #for e in e_rot:
               # print(e.item())
            print(np.mean(e_rot, axis=0))
            recalls = []
            for e_th in [2, 5, 10, 15, 25, 40]:
                recalls.append(calc_erot_recall(e_rot, e_th))
                name = e_file.split('\\')[-1]
            string = f"{task}: {name}: AR: {np.round(np.array(recalls).mean(), 6)}"
            print(string)
            with open(res_file, 'a') as f:
                f.write(string)


if __name__ == "__main__":
    main()
