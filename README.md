# SARR: Symmetry-Aware Rotation Representation

## Publication 

Paper: [IJCV (open access)](https://doi.org/10.1007/s11263-026-02770-x), [arXiv](https://arxiv.org/abs/2604.18208), [PDF](IJCV_Version_of_Record.pdf)

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

## Visualizer
You can use the newer and faster THREE.js-based [interactive visualizer](https://akriegler.github.io/SARR/visualizer/SARR_viewer_v2.html) to explore and better understand the SARR representation. For details see the paper.

For the old (unmaintained) visualization toolkit used to create the figures from the paper, open [this page](https://akriegler.github.io/SARR/visualizer/old/SARR_viewer_v1.html) which uses [pre-rendered .html's](visualizer/old/renders) or run [visualization.py](visualizer/old/visualization.py) yourself to customize it. A video explaining the plots for T-LESS symmetry class II is available [here](visualizer/old/supplementary_video_visualization-toolkit_T-LESS_symmetry_II.mp4).

## SARR mapping 
### Setup
Create a virtual environment and install dependencies:
```
$ python -m venv SARR_env
$ source ./SARR_env/bin/activate   (LINUX) or
$ .\SARR_env\Scripts\activate     (WINDOWS)
$ pip install -r requirements.txt
```

###  T-LESS/ITODD ground-truth SARR mapping 
Implementation of the SARR representation and its inverse mapping is available in [SARR.py](source/SARR.py).

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

A(M)GPD is only supported for the T-LESS dataset. The A(M)GPD calculation in [amgpd.py](source/metrics/amgpd.py) is our own, non-Torch reimplementation of [this script](https://github.com/GANWANSHUI/ES6D/blob/master/lib/tless_gadd_evaluator.py).

To reproduce our AR_B scores use the [bop-toolkit](https://github.com/thodan/bop_toolkit), specifically <eval_bop19_pose.py> (with ground-truth translation).

**NOTE (2026/05/06)**: We have added additional failsafes for edge-cases during the SARR representation mapping step, relevant for 7/123 instances (ITODD) and 16/6423 (TLESS). This results in slightly different ground-truths, available as *_v2.csv-files. Metric scripts described above still call original ground-truths, as was done for the results in paper. 

### Custom dataset SARR mapping & evaluation
You can try out mapping your own orientation predictions and ground-truth for your own dataset using the examples provided in the [example folder](example). The tool expects results to be in a format very similar to BOP, although it assumes column-major flattening, see [gt](example/gt.csv) and [pred](example/prediction.csv) files.

Information regarding the objects, i.e. their name and symmetry-class, go to the [dataset definitions](example/example_dataset_definitions.py).

Then, run
```
$ python -m example.example_mapping
```
 to map to canonic rotations through SARR.
 
The AR_C metric can then be calculated using
```
$ python -m example/example_evaluation.py
```

<br>
<br>

**LICENSE:** [CC BY 4.0](LICENSE)
