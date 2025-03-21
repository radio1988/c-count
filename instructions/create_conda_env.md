# Instructions on creating the conda environment 

## Install miniconda or conda

Follow [instructions](https://www.anaconda.com/docs/getting-started/miniconda/install) to install miniconda or conda


## Automatically create ccount-env using yml

We need to create a conda environment for C-COUNT, which includes all the python packages that C-COUNT depends on.
We can install it using the provided [ccount-env.yml](https://raw.githubusercontent.com/radio1988/ccount/refs/heads/master/workflow/env/ccount-env.yml) file, which is located in the `workflow/env` directory of the C-COUNT github repository.

```commandline
conda env create -f ccount-env.yml
```


## Manual Install

Sometimes, on some computers, especially on macbook with M processors. And installing all packages listed in ccount-env.yml one by one may be necessary to avoid package version conflicts.

```bash
conda create -n ccount-env python=3.8  # 3.8.20, 2024/09
conda activate ccount-env
conda config --add channels anaconda
conda config --add channels bioconda
conda config --add channels conda-forge
conda install keras tensorflow 
# The following is typically installed after keras and tensorflow got installed
# hdf5-1.10.6, h5py-3.1.0, numpy-1.19.5,
# cudatoolkit-11.7.0, cudnn-8.2.1.32,
# tensorflow-base-2.6, nccl-2.12.12.1, scipy-1.10.1, python-3.8.15, keras 2.6
conda install scikit-learn  # keras and tf downgraded to 2.4.3 (2020), h5py to 2.1
conda install anaconda::scikit-image #skimage
conda install matplotlib pandas
conda install aicsimageio
conda install jupyterlab ipython
conda install opencv
conda install imgaug
conda install snakemake
conda install aicspylibczi>=3.0.5 
conda install seaborn
```

## Test installation

### Test importing packages

```commandline
conda activate ccunt-env
python workflow/scripts/test_import.py
```

### Test Run on Example Data

The Example data are very small subsets of the training data used in the C-COUNT paper. Also, to ensure quick testing, we have provided a very quick setting for training. So the classification and counting performance of the example data is horrible and does not reflect the performance of C-COUNT on real data.

```commandline
snakemake -j1 reset -s workflow/train.Snakefile
```

Here are the testing resources:
/pi/merav.socolovsky-umw/rui/paper_results/install/1_training_test (10mins run)
/pi/merav.socolovsky-umw/rui/paper_results/install/2_counting_test (10mins run)
