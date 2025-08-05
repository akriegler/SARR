import csv
import yaml

import numpy as np

from easydict import EasyDict as edict
from utils import easydict_constructor

from dataset_definitions import TLESS_OBJECTS
from sym_aware_representation import map_R_to_canonic_R


def main(f_path, f_mod_path, test_targets):
    header = "scene_id,im_id,obj_id,score,R,t,time"
    with open(f_path, "r") as f, open(f_mod_path, "w", newline="") as f_mod:
        writer = csv.writer(f_mod, delimiter=',')
        line_id = 0
        prev_obj_id = ''
        prev_img_id = ''
        inst_count = 1
        for line in f:
            line_id += 1
            elems = line.split(',')
            scene_id = elems[0]
            img_id = elems[1]
            obj_id = elems[2]
            if line_id == 1 and header in line:
                writer.writerow([elem for elem in header.split(',')])
                continue
            elif f'{scene_id}-{img_id}-{obj_id}' in test_targets:
                if prev_obj_id == obj_id and prev_img_id == img_id:
                    inst_count += 1
                    if inst_count > test_targets[f'{scene_id}-{img_id}-{obj_id}']:
                        prev_obj_id = obj_id
                        prev_img_id = img_id
                        continue
                else:
                    inst_count = 1
                prev_obj_id = obj_id
                prev_img_id = img_id
                R = elems[4]
                R = np.array([float(entry) for entry in R.split(' ')]).reshape(3, 3)
                sym_v = TLESS_OBJECTS[int(obj_id)]['sym_v']
                R_new = map_tless_R_to_canonic_tless_R(R, sym_v, clamp=True)
                flattened_matrix = [str(np.round(entry, 8)) for row in R_new for entry in row]
                R_new_str = ' '.join(flattened_matrix)
                elems[4] = R_new_str
                new_line = ','.join(elems)
                writer.writerow([entry for entry in new_line.split(',')])



# As the T-LESS (BOP) annotations are ambiguous in rotation due to symmetry, this script maps them to a canonic rotation
if __name__ == '__main__':
    tless_path = 'NULL'
    with open(tless_path + r'\base\test_targets_bop19.json', 'r') as f:
        yaml.add_constructor('tag:yaml.org,2002:python/object/new:easydict.EasyDict', easydict_constructor)
        test_targets_raw = yaml.load(f, Loader=yaml.FullLoader)

    test_targets = edict()
    for si in range(1, 21):
        si = str(si)
        test_targets[si] = edict()
        for ii in range(0, 502):
            ii = str(ii)
            test_targets[si][ii] = edict()
            for oi in range(1, 31):
                oi = str(oi)
                test_targets[si][ii][oi] = 0

    for entry in test_targets_raw:
        test_targets[f'{entry["scene_id"]}-{entry["im_id"]}-{entry["obj_id"]}'] = entry['inst_count']

    f_path = tless_path + r'\results\gt\gt_tless-test.csv'
    f_mod_path = tless_path + r'results\gt\gt_tless_bop19_canonic-test.csv'
    main(f_path, f_mod_path, test_targets)
