# C-COUNT
colony count

# Installation
## Nov 2020 
- install: `conda env create -n ccount -f env.yaml`
- update: `conda env update -n ccount -f env.yaml`
## missed
- `pip install czifile opencv-python #sklearn` # missed opencv-python
- `git clone https://github.com/DmitryUlyanov/Multicore-TSNE.git; cd Multicore-TSNE/; pip install .` # missed in env.yaml

# Usage
## Counting workflow (Nov. 30, 2020)
- mkdir $workdir && cd $workdir
- ln -s $path_data data
- ln -s $path_ccount/workflow
- conda activate ccount
- cp workflow/config.yaml ./ && vim config.yaml (edit config.yaml)
- `snakemake -j 1` or `sh submit.sh` (on HPC) 

# Snakemake workflow (czi2img and npy only)
- cd $workdir
- ln -s $path/scripts
- copy and edit config.yaml (input sample names, input format)
- see submit.sh
	- source activate ccount
	- snakemake -s scripts/Snakefile  -j 1 -np
	- snakemake -s scripts/Snakefile -j 1

## Labeling workflow (Aug. 2020)
- copy ccount.py and labeling.ipynb into the working dir, alone with a xxx.npy.gz file containing the detected blobs
- source activate ccount
- jupyter-notebook
- open labeling.ipynb and work from there

## Description of scripts
- czi2npy (blob_detection.py, edge filter included)
- czi2img (czi2img.py)
- view0 of all blobs detected
- filter_merge
- labeling
- classification

## Usage Notes
- start jupyter-notebook and work from there (filter_merge, labeling)
- work from terminal (classification)
- each blob_detection takes 18G RAM

