# CCOUNT

# Description

# Installation
- install anoconda (with jupyter-notebook installed)
- `conda create -n py36 python=3.6 anaconda`
- `source activate py36`
- `conda update -y -n base -c defaults conda`
- `conda install -y -c conda-forge imgaug opencv scikit-image jupyter `
- `conda install -c anaconda -y tensorflow keras opencv cmake`
- `pip install czifile opencv-python sklearn`
- `git clone https://github.com/DmitryUlyanov/Multicore-TSNE.git; cd Multicore-TSNE/; pip install .`


# Usage
## Labeling
- copy ccount.py and labeling.ipynb into the working dir, alone with a xxx.npy.gz file containing the detected blobs
- source activate py35
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
