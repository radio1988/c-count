# CCOUNT

# Description

# Installation
## Nov 2020 
- install: `conda env create -n ccount -f env.yaml`
- update: `conda env update -n ccount -f env.yaml`
## missed
- `pip install czifile opencv-python #sklearn` # missed opencv-python
- `git clone https://github.com/DmitryUlyanov/Multicore-TSNE.git; cd Multicore-TSNE/; pip install .` # missed in env.yaml

# Usage
## Workflow
- czi2npy (blob_detection.py, edge filter included)
- czi2img (czi2img.py)
- view0 of all blobs detected
- filter_merge
- labeling
- classification
## Labeling
- copy ccount.py and labeling.ipynb into the working dir, alone with a xxx.npy.gz file containing the detected blobs
- source activate ccount
- jupyter-notebook
- open labeling.ipynb and work from there
## Usage Notes
- start jupyter-notebook and work from there (filter_merge, labeling)
- work from terminal (classification)

# Snakemake workflow (czi2img and npy only)
- cd $workdir
- ln -s $path/scripts
- copy and edit config.yaml (input sample names, input format)
- see submit.sh
	- source activate ccount
	- snakemake -s scripts/Snakefile  -j 1 -np
	- snakemake -s scripts/Snakefile -j 1
- each blob_detection takes 18G RAM
