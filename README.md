# C-COUNT

## Overview	
C-COUNT, A deep learning based tool for Colony count for colony formation assays. 

In brief, C-COUNT is a deep learning-based tool for counting colonies in colony formation assays. It is designed to work with microscopic images of plates with colonies. The tool is based on a convolutional neural network (CNN) that is custom trained to detect and count colonies in images. The tool is designed to be user-friendly and requires minimal input from the user. The user needs to provide labeled images of colonies and non-colonies, and the tool will train the CNN and then use it to count colonies in new images. The tool is designed to be flexible and can be adapted to count other types of objects in images, as long as the objects on the images are overall separated from each other.

If your lab is scoring CFU-e, you can simply use C-COUNT with the trained weigths we provided as shown in the paper.

## How it works:
- The user provides raw images from the microscope and run the blob_detection script to detect blobs (objects) in the images
- The user labels positives colonies in the images by adding an orange dot within the circle of the detected blobs
- The user runs the jpg2npy.Snakefile to convert the labeled images to npy.gz files
- The user runs the train.Snakefile to train the C-COUNT, which is a variant of LeNet-5 CNN
- The user runs the count.Snakefile with the trained weights to count colonies in new images
- Key output1 will be a table in csv format, containing the count of colonies in each image
- Key output2 will be a table in csv format, containing the size of each colony in each image (measured in num of pixels)

## More details
- All objects in the microscopic image will be detected as 'blobs', which includes colonies, undivided cells, debris, imaging artifacts, etc.
- Labeling would be needed for ccount to work. Decent classification/counting sensitivity and specificity can be achieved with 1-2 hours of labeling. When experiments are similar, we can re-use previous labeling 
- This workflow can be adapted to the counting of other biological objects, given proper labeling
- The training process takes about 30 mins on a MacbookAir with M3 processor. It takes about 2 hours on a recent 10 Core Windows desktop with CPU only. The training process only need to be done once for each type of experiment, and the trained weights can be reused for similar experiments.
- The counting process takes about 5 mins for a plate with 4 scenes on a windows desktop. The counting process can be done in parallel for multiple plates, if you have multi-core CPU. 
- All workflows are all managed by `Snakemake` , which is a workflow management system that enables reproducible and scalable analyses. If a run was terminated because of wrong input file, full-disk, out-of-RAM, user decision, etc., when you fix the issue and re-run the workflow, it will pick up from where it partially finished. It can run efficiently on high-performance computing (HPC) clusters or local machines, utilizing all available resources (e.g. CPU cores) for parallel processing.

## Using C-COUNT
### Installation
#### Prepare conda environment 
C-COUNT is designed to work with Python 3.8 and above, and requires the following packages listed in `workflow/env/mac.yaml`

If you are familiar with `conda` package manager, you can create a conda environment with the required packages by installing the packages listed in `workflow/env/mac.yaml`. Alternatively, you can use `mamba`, which is a faster alternative to `conda` for package management. Here is the example installation command (assume you have conda installed): `conda create -n ccount-env python=3.8` and then `conda env update -n ccount-env -f workflow/env/mac.yaml`. If it had package version conflicts on your machine, you can try installing each package manually one-by-one for all the packages listed in `workflow/env/mac.yaml`, with commands like this `conda install -n ccount-env -c conda-forge opencv=4.7.0` 

However, in some computers, `conda` or `mamba` will have package version conflict issues. So, we recommend using the provided `singularity` and `docker` images for easier installation and reproducibility. The `singularity` image is mainly for HPC users, while the `docker` image is for local users on a windows/mac/linux machine. 

The `docker` image is available at [docker-instruction](https://github.com/hukai916/Dockerfile_collection/tree/main/miniconda3_ccount), and the `singularity` recipe is available at [singularity-recipe](https://github.com/hukai916/Singularity_recipe_collection/tree/main/ccount), singularity image is available at [singularity-image](https://www.dropbox.com/scl/fi/qm0vwommxek33x8172j6i/ccount_gpu.sif?rlkey=iffttjocvqyupe2qmolzdhxiv&dl=0).

#### Download C-COUNT
You can download the C-COUNT repository from GitHub using the following command:
```bash
git clone 
```


### Running workflow
C-COUNT is designed to be run in a terminal interface, and all the scripts are managed by `snakemake`. The following is a brief description of how to run the workflow.

#### 1. Prepare your data
```bash
- put 


```


```bash





```bash

# Installation
- install mini-conda: https://docs.conda.io/en/latest/miniconda.html
- install mamba: https://anaconda.org/conda-forge/mamba  # optional
- install ccount-env: `mamba env create -n ccount-env -f workflow/env/mac.yaml`  # only tested in mac with M2 processer, works on Ubuntu with minor adaptations, docker image will be provided for easier installation

# Usage
## Labeling
- run `czi2img.py` on raw image and get jpg images with blobs detected (circles)
- use image editting software, e.g. 'preview', to put a red dot in all circles with a 'positive' colony.
- all blobs (circles) without red dots within will be treated as 'negatives'
- put raw and labeled images in the correct folder and run `jpg2npy.Snakefile`, e.g.  `snakemake -s workflow/jpg2npy.Snakefile -pk -j1`
- collect the npy.gz files (contains labeled blobs)

## Labeling adjustments (for advanced users)
- `labeling.ipynb`: adjusts labeling
- `crops_merging.py`: merge label files

## Training
- setup `config.train.yaml` and `config.yaml`
- run `train.Snakefile`, e.g. `snakemake -s workflow/train.Snakefile -pk -j1`
- collect `weight.hdf` for future use

## Counting
```
- mkdir $workdir && cd $workdir
- ln -s $path_data data
- ln -s $path_ccount/workflow
- conda activate ccount-env
- cp workflow/config.yaml ./ && vim config.yaml (edit config.yaml)
- cp workflow/config.data_curve.yaml ./ && vim config.data_curve.yaml 
- `snakemake -s workflow/count.Snakefile -pk -j 1` or `sh submit.sh` (on HPC)
```

## Brief Description of Key Scripts
- czi2img: read raw czi image, perform blob detection, save jpg image with detected blobs in gray circles
- crops_filtering.py: filter blob crops based on the labeling
- crops_merging.py: merge blob crop files
- get_blob_statisticss.py: show statistics on crop files, especially the labels
- crops_viewing.py: view random samples from crop files

## Notations:
- plate: each plate loaded with cells and colonies
- raw image: each czi image that is generated by the microscope, usually combining one or several scenes from the plate
- scenes: in a raw image, there are typically four scenes, which are basically a scanned square areas of a plate. Usually, there are four scenes ("Top", "Left", "Right", "Bottom") in a plate. Sometimes only one scene is imaged.
- blobs: each object detected in an image, saved in npy format, each row is `[y, x, r, label]`
- crops: a square image with the blob in the center, saved in npy format, each row is `[y, x, r, label, area, place_holder, flattened cropped_blob_img]`

## Lab Tips:
- connect to workstation via ssh (terminal interface):
	- turn on jupyter-lab on the lab workstation
	- `ssh -L3333:localhost:8888  socolovsky_lab@IP` on your laptop
 	- go to `localhost:3333` in default browser to access jupyter-lab from your laptop

