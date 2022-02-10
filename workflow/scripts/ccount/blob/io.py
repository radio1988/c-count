def load_locs(fname):
    '''
    assuming reading fname.locs.npy.gz
    read into np.array
    '''
    import gzip
    import os
    import numpy as np

    if os.path.isfile(fname):
        if fname.endswith('npy'):
            array = np.load(fname)
        elif fname.endswith('npy.gz'):
            f = gzip.GzipFile(fname, "r")
            array = np.load(f)
        else:
            raise Exception ("suffix not npy nor npy.gz")
        print(fname, array.shape)
        return array
    else:
        raise Exception ('input file', fname, 'not found')


def load_crops(in_db_name, n_subsample=False, seed=1):
    '''
    use parameters: db_name
    input: db fname from user (xxx.npy)
    output: array (crops format)
    '''
    import gzip
    import os
    import numpy as np
    from .misc import crops_stat, crop_width, sub_sample

    if not os.path.isfile(in_db_name):
        raise Exception("{} file not found".format(in_db_name))

    print("Reading {}".format(in_db_name))
    if in_db_name.endswith('npy'):
        image_flat_crops = np.load(in_db_name)
    elif in_db_name.endswith('npy.gz'):
        f = gzip.GzipFile(in_db_name, "r")
        image_flat_crops = np.load(f)
    else:
        raise Exception ("db suffix not npy nor npy.gz")

    if n_subsample:
        print("subsampling to", n_subsample, "blobs")
        image_flat_crops = sub_sample(image_flat_crops, n_subsample, seed=seed)  

    print("n-crop: {}, crop width: {}".\
        format(len(image_flat_crops), crop_width(image_flat_crops)))
    crops_stat(image_flat_crops)
    
    return image_flat_crops


def save_crops(crops, fname):
    import subprocess, os
    import numpy as np
    from .misc import crops_stat, crop_width
    from pathlib import Path
    print("Saving", fname)
    Path(os.path.dirname(fname)).mkdir(parents=True, exist_ok=True)
    crops_stat(crops)
    print('crops dim:', crops.shape)
    print('crop width:', crop_width(crops))
    fname = fname.replace(".npy.gz", ".npy")
    np.save(fname, crops)
    subprocess.run("gzip -f " + fname, shell=True, check=True)


def save_locs(crops, fname):
    import subprocess, os
    import numpy as np
    from .misc import crops_stat, crop_width
    from pathlib import Path
    print("Saving", fname)
    Path(os.path.dirname(fname)).mkdir(parents=True, exist_ok=True)
    print('crops dim:', crops.shape)
    fname = fname.replace(".npy.gz", ".npy")
    np.save(fname, crops)
    subprocess.run("gzip -f " + fname, shell=True, check=True)

