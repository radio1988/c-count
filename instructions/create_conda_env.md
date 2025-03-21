# Creating the conda environment and running C-COUNT

## Install miniconda or conda

Follow [instructions](https://www.anaconda.com/docs/getting-started/miniconda/install) to install miniconda or conda


## Download C-COUNT from github
You can download the C-COUNT repository from GitHub using the following command:

```bash
git clone https://github.com/radio1988/ccount.git
```

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
# hdf5, h5py, numpy, nccl, scipy
conda install scikit-learn  
conda install anaconda::scikit-image 
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

This help confirm all the python packages used by C-COUNT was installed.

```commandline
conda activate ccunt-env
python workflow/scripts/test_import.py
```

### Test Run on Example Data

This test run is to ensure that the C-COUNT workflow is working correctly with a very small example dataset, with a very 'quick and dirty' training setting. As a result, the classification and counting performance of the example data is horrible and does not reflect the performance of C-COUNT on real data.


### The training workflow
Pre-requisite:
- ccount github repository is downloaded as `ccount` folder (`git clone https://github.com/radio1988/ccount.git`)
- do not copy `1_training_test` foler to another location for testing

```commandline
cd ccount/resources/test_runs/1_training_test

conda activate ccount-env
snakemake -s workflow/train.Snakefile  -j1  reset  # remove all previous results 
snakemake -s workflow/train.Snakefile -pk --ri -j1  # run training on example data (this may take 2-10 mins depending n your computer)
snakemake -s workflow/train.Snakefile -j 1 --report report.html  # generate report on the run-time

# -p: print out the shell commands that will be run
# -k: keep running even if some rules fail
# --ri: re-run all the rules that have been run before
# -j1: run one job at a time, useful for debugging, you can set -j4 or -j16 for real jobs if your computer can handle the RAM usage
```

###  The counting workflow
Pre-requisite: 
- download the example czi file from [here](https://www.dropbox.com/scl/fi/1zqazamukd5i69ers9mns/1unitEpo_1-Stitching-01.czi?rlkey=z1mxzxdk1tr4si2buxhk5kkyk&dl=0), and put it in `ccount/resources/test_runs/2_counting_test/data/czi/` folder
- have to run this after the training workflow finishes, so that the trained weight h5 file is generated
- do not copy `2_counting_test` foler to another location for testing

```commandline
cd ccount/resources/test_runs/2_counting_test

conda activate ccount-env
snakemake -s workflow/count.Snakefile -j1 reset  # remove all previous results
snakemake -s workflow/count.Snakefile -pk --ri -j1  # run counting on example data
snakemake -s workflow/count.Snakefile -j 1 --report report.html  # generate report on the run-time
```
