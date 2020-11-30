pigz -1 -p 4 *npy

jupyter nbconvert --to html x.ipynb

jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace x.ipynb


