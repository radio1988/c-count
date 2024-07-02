import gzip, os, subprocess
import numpy as np
from .misc import crops_stat, crop_width


def load_locs(fname):
    '''
    assuming reading fname.locs.npy.gz
    read into np.array
    can also read crops without giving statistics
    x,y,z,L,crop-flatten
    '''
    if os.path.isfile(fname):
        if fname.endswith('npy'):
            array = np.load(fname)
        elif fname.endswith('npy.gz'):
            f = gzip.GzipFile(fname, "r")
            array = np.load(f)
        else:
            raise Exception("suffix not npy nor npy.gz")
        print(fname, array.shape)
        if array.shape[1] > 3:  # if with label
            crops_stat(array)
        if array.shape[1] > 4:
            print("n-crop: {}, crop width: {}".format(len(array), crop_width(array)))
        return array
    else:
        raise Exception('input file', fname, 'not found')


def load_crops(in_db_name):
    '''
    alias for load_locs
    '''
    image_flat_crops = load_locs(in_db_name)
    return image_flat_crops


def save_locs(crops, fname):
    from .misc import crops_stat, crop_width
    from pathlib import Path
    crops = crops[:, 0:4]
    Path(os.path.dirname(fname)).mkdir(parents=True, exist_ok=True)
    print('dim:', crops.shape)
    if crops.shape[1] > 4:
        print('width:', crop_width(crops))
        print("Saving crops:", fname)
    else:
        print("Saving locs:", fname)
    fname = fname.replace(".npy.gz", ".npy")
    np.save(fname, crops)
    subprocess.run("gzip -f " + fname, shell=True, check=True)


def save_crops(crops, fname):
    save_locs(crops, fname)
