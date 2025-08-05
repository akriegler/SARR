import csv

import numpy as np

from dataset_definitions import ITODD_OBJECTS
from sym_aware_representation import map_R_to_canonic_R


def main(f_path, f_mod_path):
    header = "scene_id,im_id,obj_id,score,R,t,time"
    with open(f_path, "r") as f, open(f_mod_path, "w", newline="") as f_mod:
        writer = csv.writer(f_mod, delimiter=',')
        line_id = 0
        for line in f:
            line_id += 1
            elems = line.split(',')
            obj_id = elems[2]
            if line_id == 1 and header in line:
                writer.writerow([elem for elem in header.split(',')])
                continue
            R = elems[4]
            t = elems[5]
            R = np.array([float(entry) for entry in R.split(' ')]).reshape(3, 3)
            t = np.array([float(entry) for entry in t.split(' ')])
            sym_v = ITODD_OBJECTS[int(obj_id)]['sym_v']
            # from utils import egocentric_to_allocentric as ego2allo
            # R = ego2allo(R, t)
            R_new, _ = map_R_to_canonic_R(R, sym_v, clamp=True)
            flattened_matrix = [str(np.round(entry, 8)) for row in R_new for entry in row]
            R_new_str = ' '.join(flattened_matrix)
            elems[4] = R_new_str
            new_line = ','.join(elems)
            writer.writerow([entry for entry in new_line.split(',')])


# As the ITODD (BOP) annotations are ambiguous in rotation due to symmetry, this script maps them to a canonic rotation
if __name__ == '__main__':
    itodd_path = r'F:\IJCV\itodd'
    f_path = itodd_path + r'\results\gt\gt_itodd-val.csv'
    f_mod_path = itodd_path + r'\results\gt\gt_itodd_bop19_canonic-test.csv'
    main(f_path, f_mod_path)
