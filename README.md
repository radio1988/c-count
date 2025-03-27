# C-COUNT

## Overview	
C-COUNT, A deep learning based tool for Colony count for colony formation assays. 

In brief, C-COUNT is a deep learning-based tool for counting colonies in colony formation assays. It is designed to work with microscopic images of plates with colonies. The tool is based on a convolutional neural network (CNN) that is custom trained to detect and count colonies in images. The user needs to provide labeled images of colonies and non-colonies, and the tool will train the CNN and then use it to count colonies in new images. The tool is designed to be flexible and can be adapted to count other types of objects in images, as long as the objects on the images are overall separated from each other.

If your lab is scoring CFU-e colonies in the hematology field, you can simply use C-COUNT with the trained weight h5 file we provided as shown in the paper.

If your lab is scoring colonies in other fields, you can ask a question in [issues](https://github.com/radio1988/c-count/issues) page and we will guide you to adapt C-COUNT to your specific needs, if your image is suitable (clear, and with well separated cells colonies). 

## How it works:
- The user provides raw images from the microscope and run the blob_detection script to detect blobs (objects) in the images, and create jpg images with detected blobs in gray circles ready for labeling
- The user labels positives colonies in the images by adding orange dots within the circle of the detected blobs [example_jpg]()
- C-COUNT uses Snakemake workflow management system to make the workflow reproducible and scalable
- The user runs the jpg2npy workflow to convert the labeled jpg images to npy.gz files
- The user runs the train workflow to train the C-COUNT, which is a variant of LeNet-5 CNN, and outputs a trained weight h5 file
- The user runs the count workflow with the trained weight h5 file to count colonies in new images, this step takes new raw images (e.g. czi files) as input
- Key output1 will be a table in csv format, containing the count of colonies in each image
- Key output2 will be a table in csv format, containing the size of each colony in each image (measured in num of pixels)
- Many other outputs will be generated, including the visualization of colony counting, distribution of colony sizes for each image etc.

## More details
- All objects in the microscopic image will be detected as 'blobs', which includes colonies, undivided cells, debris, imaging artifacts, etc.
- Labeling would be needed for ccount to work. Decent classification/counting sensitivity and specificity can be achieved with 100 positive colonies and large amount of negative colonies (important). For CFU-e, this is about 1 hours of labeling. When experiments are similar, we can re-use previous labeling trained weight h5 file to save effort. However, adding negative plate images from the new experiment would improve C-COUNT's performance, especially if your false positive rate is higher in the new experiment.
- This workflow can be adapted to the counting of other biological objects, given proper labeling
- The training process takes 30-45 mins on a MacbookAir with 16GB RAM and a M3 processor, about 60 mins on a desktop with  13th Gen Intel(R) Core(TM) i7-13700F CPU. The training process only need to be done once for each type of experiment, and the trained weights can be reused for similar experiments.
- The counting process takes 3-5 mins for a plate with 4 scenes on a Macbook Air with M3 processor. The counting process can be done in parallel for multiple plates, if you have multi-core CPU and enough RAM. Parallel processing is enabled by setting `-j 4` or `-j 8` in the `snakemake -s workflow/count.Snakefile --ri -pk -j N`command.
- All workflows are all managed by `Snakemake` , which is a workflow management system that enables reproducible and scalable analyses. If a run was terminated because of wrong input file, full-disk, out-of-RAM, user decision, etc., when you fix the issue and re-run the workflow, it will pick up from where it partially finished. It can run efficiently on high-performance computing (HPC) clusters or local machines, utilizing all available resources (e.g. CPU cores) for parallel processing.

## Using C-COUNT with conda

If you're familar with conda, see [create_conda_env.md](instructions/create_conda_env.md) for instructions on how to create a conda environment for C-COUNT, install the required packages, and running on the test data.


## Docker and singularity images
For easy installation, incase conda env creation run into technical issues, we have provided docker images for C-COUNT. See [docker.md](instructions/docker.md) for instructions on how to use the docker image.

We also provided singularity images if you are on a HPC cluster. See [singularity.md](instructions/singularity.md) for instructions on how to use the singularity image.

## Example dataset
From [zenodo.15086309](https://zenodo.org/records/15086309) you can find example datasets. It contains two example `czi` raw microscopic images, two example labeled `jpg` images for one scene in the corresponding `czi` file (each czi file here has 4 scenes). We also provided the training data set in `npy.gz` format that can be used to reproduce the training process mentioned in the manuscript. 


## Pre-trained weight h5 file
In [github/resources/weights](resources/weights) you can find the trained weight h5 file that can be used to count CFU-e colonies in the example dataset. 

