# Towards Symmetry-sensitive Pose Estimation: A Rotation Representation for Symmetric Object Classes

## Setup
Create a virtual environment and install dependencies. For example
```
python -m venv SARR_env
source ./SARR_env/bin/activate
pip install -r requirements.txt
```

## Usage
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


## Visualization toolkit
For the visualization and visual verification run [visualization.py](source/visualization.py)
The plots have been pre-rendered into interactive html-files and the toolkit can be found at: [View visualization](https://akriegler.github.io/SARR/)
A video of these plots with some explanations is available at: [View video](supplementary/supplementary_video_visualization-toolkit_symmetry_II.mp4)

## Example mapping
Implementation of our rotation representation, inverse mapping and the categorization of symmetry classes  are available in [sym_representation.py](source/sym_representation.py)


For questions contact Andreas Kriegler (andreas.kriegler@tuwien.ac.at)