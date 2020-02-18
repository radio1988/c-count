#!/bin/bash

# czi file names must have no spaces
for file in ../../ccount_data/E2F4_CFUe_14JUN19_stitching/*czi
do 
name=$(basename $file)
name=${name/.czi/}
echo ">>> For" $file $name
fname=$file runipy blob_detection.ipynb ${name}.ipynb  # work
jupyter nbconvert --to html ${name}.ipynb && rm -f ${name}.ipynb # report html then delete
mkdir -p report
mkdir -p blobs
mv ${name}.html report
mv ${name}.npy blobs
pigz blobs/${name}.npy &
done

