# CCOUNT

# Description

# Installation
- install anoconda (with jupyter-notebook installed)
- `conda create -n py35 python=3.5 anaconda`
- `source activate py35`
- `pip install czifile scikit-image imgaug opencv-python    tensorflow keras    cmake MulticoreTSNE`


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
