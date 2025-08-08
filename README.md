# Towards Symmetry-sensitive Pose Estimation: A Rotation Representation for Symmetric Object Classes

## Setup
Create a virtual environment and install dependencies:
```
$ python3.10 -m venv SARR_env
$ source ./SARR_env/bin/activate
$ pip install -r requirements.txt
```

## Usage
### Visualization toolkit
For the  visual verification and exploration of the SARR representation run [visualization.py](source/visu/visualization.py)

Plots for the T-LESS symmetry classes have been pre-rendered into interactive html-files, accessible via the visualization toolkit:  [SARR visualizer](https://akriegler.github.io/SARR/)

A video showcasing these plots for symmetry class II and giving a high-level explanation is available at: [Plot video](supplementary/supplementary_video_visualization-toolkit_T-LESS_symmetry_II.mp4)


###  Example mapping
Implementation of the SARR representation and inverse mapping are available in [sym_aware_representation.py](source/SARR/sym_aware_representation.py)

For example, we used SARR to calculate symmetry-resolved (canonic) rotation matrices as ground-truths:
```
$ python -m source.utils.tless_gt_mapping
```
or
```
$ python -m source.utils.itodd_gt_mapping
```

### Reproducing results
Set the path to the datasets and run 
```
$ python -m source.metrics.cosine
```
for the AR_C metric or 
```
$ python -m source.metrics.amgpd
```
for the AR_G metric (only T-LESS is supported).

To reproduce the AR_B scores use the [bop-toolkit](https://github.com/thodan/bop_toolkit), specifically <eval_bop19_pose.py> therein.


For questions contact Andreas Kriegler (andreas.kriegler@tuwien.ac.at)