# czi file names must have no spaces

for file in ../test/*czi ../CFUe_P8-9/*czi
do 
echo $file
fname=$file runipy blob_detection.ipynb ${file}.ipynb 1> $file.log 2> $file.err
jupyter nbconvert --to html ${file}.ipynb
done

