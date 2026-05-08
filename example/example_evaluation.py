import os
import numpy as np

from source.metrics.cosine import get_erot_matches, calc_erot_error, calc_erot_recall


def unpack_csv_gt(file, task):
    header = "scene_id"
    with open(file, "r") as f:
        rotations = {}
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
            if line_id == 1 and header in line:
                continue
            elif obj_id == prev_obj_id and prev_img_id == img_id and task == 'siso':
                continue
            prev_obj_id = obj_id
            prev_img_id  = img_id
            R = np.array(list(map(float, elems[4].split())), np.float64).reshape((3, 3), order='F')
            id = np.array(list(map(float, elems[2].split())), np.int32)
            try:
                rotations[f'{scene_id}-{img_id}-{obj_id}'].append(R)
                class_ids[f'{scene_id}-{img_id}-{obj_id}'].append(id)
            except KeyError:
                rotations[f'{scene_id}-{img_id}-{obj_id}'] = []
                rotations[f'{scene_id}-{img_id}-{obj_id}'].append(R)
                class_ids[f'{scene_id}-{img_id}-{obj_id}'] = []
                class_ids[f'{scene_id}-{img_id}-{obj_id}'].append(id)

    return rotations, class_ids


def unpack_csv_pred(file, gt_rotations, task):
    header = "scene_id"
    with open(file, "r") as f:
        rotations = {}
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
            if line_id == 1 and header in line:
                continue
            elif obj_id == prev_obj_id and prev_img_id == img_id and task == 'siso':
                continue
            if f'{scene_id}-{img_id}-{obj_id}' in gt_rotations:
                R = np.array(list(map(float, elems[4].split())), np.float64)
                cls_id = np.array(list(map(float, elems[2].split())), np.int32)
                if R.size > 1:
                    R = R.reshape((3, 3), order='F')
                else:
                    R = np.eye(3, dtype=np.float64)
                try:
                    rotations[f'{scene_id}-{img_id}-{obj_id}'].append(R)
                    class_ids[f'{scene_id}-{img_id}-{obj_id}'].append(cls_id)
                except KeyError:
                    rotations[f'{scene_id}-{img_id}-{obj_id}'] = []
                    rotations[f'{scene_id}-{img_id}-{obj_id}'].append(R)
                    class_ids[f'{scene_id}-{img_id}-{obj_id}'] = []
                    class_ids[f'{scene_id}-{img_id}-{obj_id}'].append(cls_id)
                prev_obj_id = obj_id
                prev_img_id = img_id

    return rotations, class_ids


def main():
    pred_fname = 'prediction_canon'
    gt_file = os.path.join(os.getcwd(), 'gt_canon.csv')
    pred_file = os.path.join(os.getcwd(), f'{pred_fname}.csv')
    res_file = 'AR_Cosine.txt'

    with open(res_file, 'w') as f:
        f.write(pred_fname + '\n')
        string = '-------------------------------------'
        f.write(string + '\n')
        for task in ['siso', 'vivo']:
            gt_rotations, _ = unpack_csv_gt(gt_file, task)
            pd_rotations, _ = unpack_csv_pred(pred_file, gt_rotations, task)

            gt_rotations, pd_rotations = get_erot_matches(gt_rotations, pd_rotations)
            e_rot = calc_erot_error(gt_rotations, pd_rotations)

            recalls = []
            for e_th in [2, 5, 10, 15, 25, 40]:  # thresholds taken from DOI: 10.1109/IRC55401.2022.00040
                recalls.append(calc_erot_recall(e_rot, e_th))

            ar = 100 * np.round(np.array(recalls).mean(), decimals=4)
            res_string = f"{task}: AR_C: {ar:.2f}"
            print(res_string)
            res_string_2 = f"{task}: mean-e_rot: {np.mean(e_rot):.2f}°"
            print(res_string_2)
            f.write(res_string + '\n')
            f.write(res_string_2 + '\n')
        string = '-------------------------------------'
        f.write(string + '\n')
        f.write('\n')


if __name__ == "__main__":
    print('Dont sort CSV result files for this metric. This can skew SiSo scores since only the first instance per image is considered (assumed to be most visible, from prior sorting).')
    main()
