# Model-Based Visual Planning with Self-Supervised Functional Distances 
Code release for [Model-Based Visual Planning with Self-Supervised Functional Distances](https://arxiv.org/abs/2012.15373).

This repository includes the _distance learning_ component of the MBOLD method. To learn video prediction models, we use 
the training code located at [RoboNet](https://github.com/s-tian/RoboNet/tree/mbold_release), and to run control, we use the [visual foresight](https://github.com/s-tian/visual_foresight/tree/mbold_release)
codebase. In order to run MBOLD, you will need to use all three components. If you only want to perform distance learning,
this repository will be sufficient.

## Installation:

To install the package, clone the repository and run `pip install -e .` from the main directory containing `setup.py`.

Install Meta-World: Please see the instructions [here](https://github.com/rlworkgroup/metaworld) to install Meta-World to use the simulated
environments.

Installing FAISS: To perform GPU-enabled negative training example sampling, this code uses the [FAISS](https://github.com/facebookresearch/faiss) library. Please
see [this document](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md) for information about how to install the GPU version 
of FAISS. 

## Examples

### Training distance functions
To train a distance function, run 
```
python classifier_control/train.py <path_to_experiment_config>
```
For example, to train the MBOLD distance function on the three object tabletop pushing task data, run
```
python classifier_control/train.py experiments/distfunc_training/q_func_training/tabletop_3obj/conf.py
```

### Running data collection & control experiments

To run control experiments or perform data collection, run 
```
python run_control_experiment.py <path_to_experiment_config>
```

For example, to run MBOLD on the three object tabletop pushing task, run
```
python run_control_experiment.py experiments/control/tabletop_3obj/mbold/hparams.py
```
Note that running control experiments requires a trained video prediction model, a trained distance function, as well as a set of evaluation tasks (possibly downloaded from 
the below section).

## Datasets 

We provide training datasets for each simulated environment (collected using a random policy) as well as evaluation tasks 
used in comparisons in the paper [here.](https://drive.google.com/drive/folders/13DusXI-94_5l_iQXw_Q-ylrdjrh3Wh_s?usp=sharing)
To train on these datasets, download the .zip file and unzip it in your preferred location. Then, set the parameter `'data_dir'`
in the training configuration file to point to this directory.

## License
With the exception of the Frank door environment, the code is licensed under the terms of the MIT license.
The assets from the Franka door environment are from the the playroom environment, licensed by Google Inc. under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/). The Franka Panda model is by Vikash
Kumar and licensed under the terms of the Apache 2.0 license.

## Citation
If you find this useful, consider citing:
```
@inproceedings{tian2021mbold,
  title={Model-Based Visual Planning with Self-Supervised Functional Distances},
  author={Tian, Stephen and Nair, Suraj and Ebert, Frederik and Dasari, Sudeep and Eysenbach, Benjamin and Finn, Chelsea and Levine, Sergey},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
