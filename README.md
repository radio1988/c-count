# CCOUNT

# Description

# Installation
- install anoconda
- install jupyter-notebook
- install packages
    - `pip install czifile scikit-image keras tensorflow pandas numpy matplotlib imgaug`


# Usage
1. blob_detection
2. filter_merge
3. labeling
4. classification



# Todo:
## Block Edge blobs
- Blob detector detects false positives around edges when blocks are merged
    - seperate blocks and discard edge blobs can solve this
    - adjusting blob_detector parameters can't, discards true positives
- To rescue edge blobs, and avoid double counting, need to remove blobs that overlaps, but it's a pain
- need to use branches in github to save the current progress on whole image processing

## Speed
- scaled version for detection, original for cropping blobs
- 

