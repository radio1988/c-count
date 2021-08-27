#import ccount
from ccount.blob.area_calculation import area_calculation
from ccount.blob.io import load_crops, parse_blobs
import sys
import numpy as np
import matplotlib.pyplot as plt
import subprocess


# no filtering, all reasults saved, bug negative ones should have -1 as output? as the calculation can be wrong for negative ones?? Now it is still calculated for testing


print("python area_calculation.py blob.npy.gz output.area.txt")
if len(sys.argv) is not 3:
    sys.exit("cmd error")
print("only calculate area accurately for positive blobs, negative blobs area calculation not accurate if the blob shape not regular")
print("cmd:", sys.argv)

crops = load_crops(sys.argv[1])
print(crops.shape[0], "blobs loaded ")

def area_calculation_of_blobs(crops, 
                              out_txt_name = "blobs.area.txt", 
                              title="Blob Area In Pixcels", label_filter=1,
                              plotting=True, txt_saving=True, crop_saving=True):
    '''only calculate for positive blobs'''
    images, labels, rs = parse_blobs(crops)
    print("labels", [ str(int(x)) for x in labels])
    print("label_filter", str(int(label_filter)))
    # filter
    neg_idx = [str(int(x)) != str(int(label_filter)) for x in labels]
    print("idx", neg_idx)
    # cal
    areas = [area_calculation(image, r=rs[ind], plotting=False) for ind, image in enumerate(images)]
    crops[:, 4] = areas
    crops[neg_idx, 4] = -1
    areas = crops[:, 4]

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

