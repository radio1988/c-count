#import ccount
import sys
import numpy as np
import matplotlib.pyplot as plt
from ccount import area_calculation, load_blobs_db, parse_blobs


print("python area_calculation.py blob.npy.gz label (0/1) output.area.txt")
print(sys.argv)
crops = load_blobs_db(sys.argv[1])

# filter
crops = crops[crops[:, 3] == int(sys.argv[2]), ]
print(crops.shape[0], "left after filtering")

def area_calculation_of_blobs(crops, 
                              out_txt_name = "blobs.area.txt", 
                              title="Blob Area In Pixcels", 
                              plotting=True, 
                              txt_saving=True):
    Images, Labels, Rs = parse_blobs(crops)
    areas = [area_calculation(image, r=Rs[ind], plotting=False) for ind, image in enumerate(Images)]
    
    if plotting:
        plt.hist(areas, 40)
        plt.title(title)
        plt.savefig(sys.argv[3]+".pdf")
        
    if txt_saving:
        np.savetxt(out_txt_name, areas)
    return (areas)

all_areas = area_calculation_of_blobs(crops, 
                                      title="All Blobs (Area in Pixcels)",
                                      out_txt_name=sys.argv[3], 
                                      txt_saving=True,
                                      plotting=True
                                     )

