import numpy as np

EXAMPLE_OBJECTS = {
    # corresponding to Figure 3 in https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_itodd.pdf
    'object_1': {
        'obj_id': 1,
        'obj_str': 'object_1',
        'kappa': np.array([1, 1, 4]),
    },
    'object_2': {
        'obj_id': 2,
        'obj_str': 'object_2',
        'kappa': np.array([1, 1, 1]),
    },
    'object_3': {
        'obj_id': 3,
        'obj_str': 'object_3',
        'kappa': np.array([2, 2, 2]),
    }
}

EXAMPLE_OBJECTS_BY_ID = {
    EXAMPLE_OBJECTS[key]['obj_id']: EXAMPLE_OBJECTS[key] for key in EXAMPLE_OBJECTS.keys()
}
