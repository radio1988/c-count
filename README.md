# CCOUNT

# Description

# Installation (manual)
- install anoconda (with jupyter-notebook installed)
- `conda create -n py36 python=3.6 anaconda`
- `source activate py36`
- `conda update -y -n base -c defaults conda`
- `conda install -y -c conda-forge imgaug opencv scikit-image jupyter ` # replaced
- `conda install -c anaconda -y tensorflow keras opencv cmake scikit-learn` # replaced
- `pip install czifile opencv-python #sklearn` # missed opencv-python
- `git clone https://github.com/DmitryUlyanov/Multicore-TSNE.git; cd Multicore-TSNE/; pip install .` # missed in env.yaml

# 2020 install
conda install tensorflow keras scikit-image czifile  runipy imgaug

# Nov 2020 install
- install: `conda env create -n ccount -f env.yaml`
- update: `conda env update -n ccount -f env.yaml`


# Usage
## Labeling
- copy ccount.py and labeling.ipynb into the working dir, alone with a xxx.npy.gz file containing the detected blobs
- source activate py36/tf
- jupyter-notebook
- open labeling.ipynb and work from there


## Usage Notes
- start jupyter-notebook and work from there (blob_detection, filter_merge, labeling)
- work from terminal (classification)

# Workflow
1. blob_detection
2. filter_merge
3. labeling
4. classification

# Snakemake workflow (czi2img and npy only)
- copy everything from scripts/ into workdir
- edit config.yaml (input sample names, input format)
- source activate ccount
- snakemake -j 1 -np
- snakemake -j 1
- each blob_detection takes 18G RAM
