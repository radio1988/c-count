import gzip, os, subprocess
import numpy as np
from .misc import crops_stat, crop_width


def load_locs(fname):
    '''
    assuming reading fname.locs.npy.gz
    read into np.array
    can also read crops without giving statistics
    y,x,r,L,crop-flatten
    '''
    if os.path.isfile(fname):
        if fname.endswith('npy'):
            array = np.load(fname)
        elif fname.endswith('npy.gz'):
            f = gzip.GzipFile(fname, "r")
            array = np.load(f)
        else:
            raise Exception("suffix not npy nor npy.gz")

        if array.shape[1] > 3:  # if with label
            print("Read locs file: ", fname, array.shape)
            crops_stat(array)
        if array.shape[1] > 4:
            print("Read crops file: ", fname, array.shape)
            print("n-crop: {}, crop width: {}\n".format(len(array), crop_width(array)))
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
    """
    Input: np.array of crops or locs
    Output: npy file (not npy.gz)

    Note:
    - if input are crops, trim to xyrL then save locs (to save space)
    - even if fname is x.npy.gz will save into x.npy (locs file not big anyway)
    """
    from .misc import crops_stat, crop_width
    from pathlib import Path

    # Trim crops to locs
    if crops.shape[1] > 4:
        locs = crops[:, 0:4]

    print('num of blob locs to save: {}'.format(locs.shape[0]))
    print('locs are in yxrL format, only {} nums per row\n'.format(locs.shape[1]))

    Path(os.path.dirname(fname)).mkdir(parents=True, exist_ok=True)

    fname = fname.replace(".npy.gz", ".npy")
    print('Saving locs:', fname)
    np.save(fname, locs)

    subprocess.run('gzip -f ' + fname, shell=True, check=True)
    print('\n')


def save_crops(crops, fname):
    """
    Input: crops
    Output: npy.gz or npy files
    """
    from .misc import crops_stat, crop_width
    from pathlib import Path
    Path(os.path.dirname(fname)).mkdir(parents=True, exist_ok=True)
    print('dim:', crops.shape)
    if crops.shape[1] > 4:
        print('width:', crop_width(crops))
        print("Saving crops:", fname)
    else:
        print("Saving locs:", fname)

    if fname.endswith('.npy.gz'):
        fname = fname.replace(".npy.gz", ".npy")
        np.save(fname, crops)
        subprocess.run("gzip -f " + fname, shell=True, check=True)
    elif fname.endswith('.npy'):
        np.save(fname, crops)
    else:
        raise Exception ('crop output suffix not .npy nor .npy.gz')
