#!/bin/bash
for f in ../data2/*/
do  
	n=$(basename $f)
	dirname=$n runipy merge.downsample.area_calcu.view.save.ipynb $n.ipynb
       	jupyter nbconvert --to html $n.ipynb
	mv $n.ipynb report
	mv $n.html report
done
