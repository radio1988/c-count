#!/bin/bash

for file in *pred.npy.gz
do
    name=$(basename $file)
    name=${name/.npy.gz/}
    echo ">>> For" $file $name
    fname=$file runipy viewing_blobs.ipynb ${name}.ipynb  # work
    jupyter nbconvert --to html ${name}.ipynb && rm -f ${name}.ipynb # report html then delete
    mkdir -p html
    mv ${name}.html html
done

