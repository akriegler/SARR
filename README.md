# Towards Symmetry-sensitive Pose Estimation: A Rotation Representation for Symmetric Object Classes

## Publication 

Paper: [IJCV (open access)](https://doi.org/10.1007/s11263-026-02770-x), [arXiv](https://arxiv.org/abs/2604.18208) 

Authors: [Andreas Kriegler](https://andreaskriegler.eu/), 
[Csaba Beleznai](https://publications.ait.ac.at/de/persons/csaba.beleznai/),
and [Margrit Gelautz](https://informatics.tuwien.ac.at/people/margrit-gelautz) 

If you use this project, please cite:

```bibtex
@article{Kriegler2026,
  title = {Towards Symmetry-sensitive Pose Estimation: A Rotation Representation for Symmetric Object Classes},
  author = {Kriegler, Andreas and Csaba, Beleznai and Gelautz, Margrit},
  journal = {International Journal of Computer Vision},
  publisher = {Springer Nature},
  year = {2026},
  volume = {134},
  number = {212},
  pages = {1--24},
  doi = {10.1007/s11263-026-02770-x},
}
```
Contact: *andreas.kriegler@tuwien.ac.at*

## Setup
Create a virtual environment and install dependencies:
```
$ python3.10 -m venv SARR_env
$ source ./SARR_env/bin/activate
$ pip install -r requirements.txt
```

## Usage
### Visualization toolkit
For the  visual verification and exploration of the SARR representation run [visualization.py](source/visu/visualization.py).

```
$ python -m source.visu.visualization
```

Plots for the T-LESS symmetry classes have been pre-rendered into interactive html-files, accessible via the visualization toolkit:  [SARR visualizer](https://akriegler.github.io/SARR/).

A video showcasing and explaining these plots for symmetry class II is available here: [explanation video](video/supplementary_video_visualization-toolkit_T-LESS_symmetry_II.mp4).


###  SARR representation mapping
Implementation of the SARR representation and its inverse mapping is available in [sym_aware_representation.py](source/sym_aware_representation.py).

For example, we mapped to SARR and back to obtain symmetry-resolved (canonic) rotation matrices as ground-truths for our experiments:
```
$ python -m source.utils.tless_gt_mapping
```
or
```
$ python -m source.utils.itodd_gt_mapping
```

### Reproducing results
To calculate additional pose estimation evaluation metrics beyond the standard BOP-scores, or to reproduce the results from our paper, set the paths in [cosine.py](source/metrics/cosine.py) and [amgpd.py](source/metrics/amgpd.py) and run
```
$ python -m source.metrics.cosine
```
for the AR_C metric or 
```
$ python -m source.metrics.amgpd
```
for the AR_G metric.

A(M)GPD is only supported for the T-LESS dataset. The A(M)GPD calculation in [amgpd.py](source/metrics/amgpd.py) is our own reimplementation of [this script](https://github.com/GANWANSHUI/ES6D/blob/master/lib/tless_gadd_evaluator.py).

To reproduce our AR_B scores use the [bop-toolkit](https://github.com/thodan/bop_toolkit), specifically <eval_bop19_pose.py> (with ground-truth translation).


## COMING SOON
Training + inference code

**MIT license**