#import ccount
import sys
import numpy as np
import matplotlib.pyplot as plt
from ccount import area_calculation, load_blobs_db, parse_blobs

# no filtering, all reasults saved, bug negative ones should have -1 as output? as the calculation can be wrong for negative ones?? Now it is still calculated for testing


print("python area_calculation.py blob.npy.gz output.area.txt")
print("only calculate area accurately for positive blobs, negative blobs area calculation not accurate if the blob shape not regular")
print("cmd:", sys.argv)

crops = load_blobs_db(sys.argv[1])
print(crops.shape[0], "blobs loaded ")

def area_calculation_of_blobs(crops, 
                              out_txt_name = "blobs.area.txt", 
                              title="Blob Area In Pixcels", 
                              plotting=True, txt_saving=True, crop_saving=True):
    Images, Labels, Rs = parse_blobs(crops)
    areas = [area_calculation(image, r=Rs[ind], plotting=False) for ind, image in enumerate(Images)]
    crops[4] = areas
    
    if plotting:
        plt.hist(areas, 40)
        plt.title(title)
        plt.savefig(sys.argv[2]+".pdf")
        
    if txt_saving:
        np.savetxt(out_txt_name, areas)

    if crop_saving:
        np.save(sys.argv[2]+".npy", crops)
        bashCommand = "gzip -f " + sys.argv[2]+".npy"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()


    return (areas)

all_areas = area_calculation_of_blobs(crops, 
                                      title="All Blobs (Area in Pixcels)\nNeg blobs have -1 as area",
                                      out_txt_name=sys.argv[2], 
                                      txt_saving=True, plotting=True, crop_saving=True
                                     )

