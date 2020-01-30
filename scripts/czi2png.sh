#!/bin/bash

# czi file names must have no spaces
for file in  ../../ccount_data/IL17a_CFUe_24JAN20/*czi
do 
name=$(basename $file)
name=${name/.czi/}
echo ">>> For" $file $name
fname=$file runipy czi2png.ipynb ${name}.ipynb  # work
jupyter nbconvert --to html ${name}.ipynb && rm -f ${name}.ipynb  && rm -f ${fname}.html # report html then delete
mkdir -p png
mkdir -p equalized_png
mv ${fname}.png  png 
mv ${fname}.equ.png equalized_png
done
