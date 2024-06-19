import gzip, os, subprocess
import numpy as np
from .misc import crops_stat, crop_width


def load_locs(fname):
    '''
    assuming reading fname.locs.npy.gz
    read into np.array
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
        return array
    else:
        raise Exception('input file', fname, 'not found')


def load_crops(in_db_name):
    '''
    use parameters: db_name
    input: db fname from user (xxx.npy)
    output: array (crops format)
    '''
    image_flat_crops = load_locs(in_db_name)

    if (image_flat_crops.shape[1] > 3):  # with label
        crops_stat(image_flat_crops)

    if (image_flat_crops.shape[1] > 4 + 4):
        print("n-crop: {}, crop width: {}". \
              format(len(image_flat_crops), crop_width(image_flat_crops)))

    return image_flat_crops


def save_crops(crops, fname):
    from .misc import crops_stat, crop_width
    from pathlib import Path
    print("Saving crops:", fname)
    Path(os.path.dirname(fname)).mkdir(parents=True, exist_ok=True)
    crops_stat(crops)
    print('dim:', crops.shape)
    print('width:', crop_width(crops))
    fname = fname.replace(".npy.gz", ".npy")
    np.save(fname, crops)
    subprocess.run("gzip -f " + fname, shell=True, check=True)


def save_locs(crops, fname):
    from .misc import crops_stat, crop_width
    from pathlib import Path
    crops = crops[:, 0:4]
    print("Saving locs:", fname)
    Path(os.path.dirname(fname)).mkdir(parents=True, exist_ok=True)
    print('dim:', crops.shape)
    fname = fname.replace(".npy.gz", ".npy")
    np.save(fname, crops)
    subprocess.run("gzip -f " + fname, shell=True, check=True)
