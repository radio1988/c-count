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

See [create_conda_env.md](instructions/create_conda_env.md) for instructions on how to create a conda environment for C-COUNT and install the required packages.
