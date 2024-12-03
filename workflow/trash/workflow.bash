#!/bin/bash

### 0_czi2png ### 
mkdir -p 0_czi2png
cd 0_czi2png/

cp ../czi2png.ipynb  ./
ln -s ../ccount.py

for file in  ../../ccount_data/Ashley_Epo_DRC_31JAN20/*czi
do
name=$(basename $file)
name=${name/.czi/}
echo ">>> For" $file $name
fname=$file runipy czi2png.ipynb ${name}.ipynb  # work
jupyter nbconvert --to html ${name}.ipynb && rm -f ${name}.ipynb  && rm -f ${fname}.html # report html then delete
mkdir -p png
mkdir -p equalized_png
mkdir -p html
mv ${name}.png  png
mv ${name}.equ.png equalized_png
mv ${name}.html html
done

cd ..


### 1_blob_detection ###
mkdir -p 1_blob_detection/
cd 1_blob_detection
ln -s ../ccount.py
ln -s ../pyimagesearch
cp ../blob_detection.ipynb .

# czi file names must have no spaces
for file in ../../ccount_data/Ashley_Epo_DRC_31JAN20/*czi
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

cd ..

### 2_classification ###
mkdir -p 2_classification_rui_daniel1
cd 2_classification_rui_daniel1
cp ../classification.bsub .
cp ../blob_classification.py .
ln -s ../ccount.py
ln -s ../pyimagesearch
cp ../viewing_blobs.ipynb .

for f in ../1_blob_detection/blobs/*npy.gz
do
# round1: pred_rui_net
weight=../../run1_good_ram_problem/2_first_round_training/output/Z_CFUe_1-Stitching-74.hdf5
echo python blob_classification.py -db $f -l 1 -w $weight
python blob_classification.py -db $f -l 1 -w $weight #> $f.classification.log 2>&1
# round2: pred_daniel_net
weight=../../run1_good_ram_problem/5_second_round_training/danielAug19.hdf5
f=$(basename $f .npy.gz)
f=${f}.yes.npy.gz
echo python blob_classification.py -db $f -l 1 -w $weight
python blob_classification.py -db $f -l 1 -w $weight > $f.pred2.log 2>&1
done

mkdir -p step1_all
mkdir -p step1_yes
mkdir -p step2_all
mkdir -p step2_yes
mv *yes.yes.npy.gz step2_yes
mv *yes.pred.npy.gz step2_all
mv *pred2.log step2_yes
mv *yes.npy.gz step1_yes
mv *pred.npy.gz step1_all
mv *pred.txt step1_yes

# count
grep "Predictions" step2_yes/*pred2.log > COUNTS.txt

# viewing
mkdir -p step2_all_html

for file in step2_all/*pred.npy.gz
do
    name=$(basename $file)
    name=${name/.npy.gz/}
    echo ">>> For" $file $name
    fname=$file runipy viewing_blobs.ipynb ${name}.ipynb  # work
    jupyter nbconvert --to html ${name}.ipynb && rm -f ${name}.ipynb # report html then delete
    mv ${name}.html step2_all_html
done

cd ..
