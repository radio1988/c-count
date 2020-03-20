## Instructions
1. blob_detection
2. size_filter
3. labeling_blobs
4. Balancing data
5. counting (classification)

## Installation
- install conda
- `conda install skimage`

## Nomenclature:
- image: the whole input czi, which contains many blocks
- block: each block of image scanned by microscopy
- blob: each blob contrasty (darker on light background) detectable in the block
- crop: a 50x50 cropped image with the blob in center, with a yellow circle indicating 
- masked_crop: 


>>> Todo: 
[√] detection of faint blobs
[] reduce RAM
[√] two steps: 
[√]    1. edge removal 
[v]     2. daniel labeling ML
[] speed up cropping after blob detection


>>> Jan14
- TSNE single core option
- updated read_czi
- jpg not supported, quick fix as png
- github code not uptodate, should use run1 code, maybe run1-6
- pip uninstall/install pillow fixed jpg not supported issue
- equalize whole image (not good, still block equalization)
- remember to visualize for batch production mode
- more sensitive blob_log detection, smaller threshold, more num_sigma, still use 4 scale for speed, 27GB RAM during blob_detection
- fix block_equalization bug for mod
- updated blob_detection.sh, save html in workdir now
- filtered edge blobs
- count new data

>>> Jan28
- removed tsne
- added viewing_blobs.sh, updated viewing_blobs.ipynb

>>> March 04
- still bash with 3 steps

>>> March 20
- czi2img new format 2019/2020 -> 2019
- snakemake workflow (cmd version, not ipynb interactive mode)
