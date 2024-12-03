import sys, subprocess
import numpy as np
import matplotlib.pyplot as plt
from ccount_utils.blob import load_blobs, save_crops
from ccount_utils.blob import area_calculations



# no filtering, all reasults saved, bug negative ones should have -1 as output? as the calculation can be wrong for negative ones?? Now it is still calculated for testing


print("usage: python <area_calculation.py>  <blob.npy.gz>  <output.area.txt>")
print("area calulation for non-colony is sometimes inaccurate, expecially when it is empty")
print("cmd:", sys.argv)

if len(sys.argv) is not 3:
    sys.exit("cmd error")

inname = sys.argv[1]
outname_txt = sys.argv[2]
outname_core = outname_txt.replace('.txt', '')
outname_hist = outname_core + '.hist.png'
outname_crops = outname_core + '.npy.gz'
print('inname:', inname, 
    "\noutname_txt", outname_txt, 
    "\noutname_crops", outname_crops)

crops = load_blobs(inname)
areas = area_calculations(crops)

# save txt
np.savetxt(outname_txt, areas, fmt='%i', delimiter='')

# save crops
crops[:, 4] = areas
save_crops(crops, outname_crops)


plt.hist(areas, 40)
plt.title(outname_core)
plt.savefig(outname_hist)
