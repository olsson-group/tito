# Transferable Generative Models Bridge Femtosecond to Nanosecond Time-Step Molecular Dynamics

## Inference
To sample a trajetory with 1 ns timesteps for a molecule in a pdb file with TITO, run:
```
python scripts/sample.py --pdb_path inference_files/test_mol.pdb --model_path inference_files/mdqm9.ckpt --data_set mdqm9 --data_path /proj/berzelius-2025-189/datasets/mdqm9/ --lag 1000 --nested_samples 10 --batch_size 1 
```

## Training
To train a TITO model, run:
```
python scripts/train.py --dataset <mdqm9 or timewarp> --data_path <path to dataset> --epochs <number_of_epochs>
```

## Installation
For the environment installation, first create an environment with python 3.11:
 ```
mamba create -n tito python=3.11
  ```
Then run:
```
  mamba activate tito
  uv pip install -e .
  uv pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
  uv pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
  uv pip install lightning torch_geometric
```

## Dataset pre-processing
### Small molecules
Download the mdqm9-nc dataset from [this link](https://github.com/olsson-group/mdqm9-nc-loaders). The h5 file, sdf file and splits need to be in the same directory.
### Timewarp
Download the Timewarp dataset using [scripts/preprocessing/download_timewarp.py](scripts/preprocessing/download_timewarp.py) and pre-process it with [scripts/preprocessing/timewarp_to_h5.py](scripts/preprocessing/timewarp_to_h5.py).

## Paper

This repository contains code to reproduce the results from:

```
@article{Diez2026,
  title = {Transferable generative models bridge femtosecond to nanosecond time-step molecular dynamics},
  volume = {12},
  ISSN = {2375-2548},
  url = {http://dx.doi.org/10.1126/sciadv.aed2333},
  DOI = {10.1126/sciadv.aed2333},
  number = {15},
  journal = {Science Advances},
  publisher = {American Association for the Advancement of Science (AAAS)},
  author = {Diez,  Juan Viguera and Schreiner,  Mathias and Olsson,  Simon},
  year = {2026},
  month = Apr 
}
```

