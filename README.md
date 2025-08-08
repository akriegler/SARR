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

Plots for the T-LESS symmetry classes and many 3D geometric primitives have been pre-rendered into interactive html-files for the visualization toolkit:  [SARR visualizer](https://akriegler.github.io/SARR/)

A video showcasing these plots for symmetry class II and giving a high-level explanation is available at: [Plot video](supplementary/supplementary_video_visualization-toolkit_symmetry_II.mp4)


###  Example mapping
Implementation of our rotation representation, inverse mapping and the categorization of symmetry classes  are available in [sym_representation.py](source/sym_representation.py)


### Reproducing results
Set the path to the datasets and run 
```
python -m source.metrics.cosine
```
for the AR_C metric or 
```
python -m source.metrics.amgpd
```
for the AR_G metric (only T-LESS is supported).

To reproduce the AR_B scores use the [bop-toolkit](https://github.com/thodan/bop_toolkit), specifically <eval_bop19_pose.py> therein.


For questions contact Andreas Kriegler (andreas.kriegler@tuwien.ac.at)