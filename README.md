# C-COUNT

## Overview: C-COUNT, a deep learning tool for measuring the number and size of erythroid progenitors. 

C-COUNT is described in 'C-COUNT: a Convolutional Neural Network-Based Tool for Automated Scoring of Erythroid Colonies' by Li, Winward, Lalonde, Hidalgo, Sardella, Hwang, Swaminathan, Thackeray, Hu, Zhu and Socolovsky (submitted). 

C-COUNT is implemented as a modified LeNet convolutional neural network (LeCun Y et al., Gradient-Based Learning Applied to Document Recognition. Proc of The IEEE. 1998). C-COUNT is trained to measure the number and size of colony-forming-unit-erythroid (CFU-e) colonies in colony-formation assays. The input to the tool are grayscale automated-microscopy bright-field images of CFU-e colony assay plates. Prior to imaging the plates are stained with diaminobenzidine, which turns CFU-e colonies dark and allows them to be distinguished from other colony types, debris or cell aggregates. 

The trained tool (provided with an h5 weights file) is expected to identify CFU-e colonies in images with a similar microscopy set-up to that described in Li et al.. It is possible to re-train the tool to improve performance for each laboratory's microscopy setup. With appropriate training datasets, it may be possible to adapt C-COUNT to identify other colony types.  

Users may ask a question in [issues](https://github.com/radio1988/c-count/issues) page to help guide them in adapting C-COUNT to their specific needs. Training dataset images need to be suitable (clear, and with well separated colonies).

## C-COUNT training workflow:

- C-COUNT uses Snakemake workflow management system to make the workflow reproducible and scalable
- A training dataset of microscopy images from CFU-e colony plates are required as input
- The 'blob-detection' script outputs the images as jpg files where objects ('blobs') are circled
- The user labels the circled blobs that are positive (CFU-e colonies) by adding an orange dot within the circle [example_jpg]()
- The user runs the 'jpg2npy' workflow to convert the labeled jpg images to npy.gz files
- The user runs the 'train' workflow, which outputs a trained weight h5 file

## C-COUNT counting workflow:

- Images of CFU-e plates from an experiment are used as input. These need to be taken with the same microscopy setup as the training dataset in the training workflow
- The user runs the 'count' workflow with the trained weight h5 file 
- Key output1 is a table in csv format, containing CFU-e colony count in each image
- Key output2 is a table in csv format, containing the size of each CFU-e colony in each image (measured in pixels)

## Pre-trained weight h5 file
The trained weight h5 file in [github/resources/weights](resources/weights) can be used to count CFU-e colonies in the example dataset. 
