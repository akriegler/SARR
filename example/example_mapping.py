import os
import csv
import numpy as np

from example.example_dataset_definitions import EXAMPLE_OBJECTS_BY_ID
from source.utils.rot_utils import map_Euler_to_R_canon, map_R_to_R_canon, map_SixD_to_R_canon, map_quat_to_R_canon, map_sarr_to_R_canon, map_trig_to_R_canon


def main(f_path, f_canon_path, rot_representation, header):
    with open(f_path, "r") as f, open(f_canon_path, "w", newline="") as f_canon:
        writer = csv.writer(f_canon, delimiter=',')
        line_id = 0
        for line in f:
            line_id += 1
            line = line.strip()
            elems = line.split(',')
            obj_id = elems[2]
            if line_id == 1 and header in line:
                writer.writerow([elem for elem in header.split(',')])
                continue
            rotation = elems[4]
            kappa = EXAMPLE_OBJECTS_BY_ID[int(obj_id)]['kappa']
            entries = np.asarray([float(entry) for entry in rotation.split(' ')])
            if rot_representation == 'euler':
                alpha = entries[0]
                beta = entries[1]
                gamma = entries[2]
                R_canon = map_Euler_to_R_canon(alpha, beta, gamma, kappa, clamp=True)
            elif rot_representation == 'rotmat':
                R = entries.reshape(3, 3, order='F')
                R_canon = map_R_to_R_canon(R, kappa, clamp=True)
            elif rot_representation == '6d':
                SixD = entries.reshape(3, 2, order='F')
                R_canon = map_SixD_to_R_canon(SixD, kappa, clamp=True)
            elif rot_representation == 'quat':
                quat = entries.reshape(1, 4, order='F')
                R_canon = map_quat_to_R_canon(quat, kappa, clamp=True)
            elif rot_representation == 'sarr':
                sarr = entries.reshape(2, 3, order='F')
                R_canon = map_sarr_to_R_canon(sarr, kappa, clamp=True)
            elif rot_representation == 'trig':
                trig = entries.reshape(2, 3, order='F')
                R_canon = map_trig_to_R_canon(trig, kappa, clamp=True)
            else:
                print('Unknown rotation representation')
                raise NotImplementedError

            flattened_matrix = list(np.round(R_canon, 8).flatten('F').astype(str))
            R_canon_str = ' '.join(flattened_matrix)
            elems[4] = R_canon_str
            new_line = ','.join(elems)
            writer.writerow([entry for entry in new_line.split(',')])


if __name__ == '__main__':
    to_map = ['gt', 'prediction']   # 'gt', 'pred'
    rot_representation = 'euler'  #  'euler', '6d', 'rotmat', 'quat', 'sarr', 'trig'. Assumptions: flattened COLUMN-WISE for 6d, rotmat, sarr and trig, intrinsic XYZ for Euler in unit radians, scalar-last (x, y, z, w) quaternion convention
    header = "scene_id,img_id,obj_id,score,rotation"

    for f_type in to_map:
        file_path = os.path.join(os.getcwd(), f'{f_type}.csv')
        canon_file_path = os.path.join(os.getcwd(), f'{f_type}_canon.csv')
        main(file_path, canon_file_path, rot_representation, header)
