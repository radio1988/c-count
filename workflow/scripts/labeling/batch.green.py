import re, os

def file_exists_safe(path):
    """
    Checks if a file exists at the given path using a try-except block for robust error handling.

    Args:
        path: The path to the file as a string.

    Returns:
        True if the file exists, False otherwise.
    """
    try:
        with open(path, 'r'):
            return True
    except FileNotFoundError:
        print(f"Warning: File '{path}' does not exist")
    return False


img_dir = '../1_ashley_label_img'
czi_dir = '../czi'
npy_dir = '../npy'

files = os.listdir(img_dir)
os.makedirs('log_green', exist_ok=True)
# goals:
# ../1_ashley_label_img/NoEpo_1-Stitching-05.0.crops.clas.npy.gz.FP.jpg
# ../czi/NoEpo_1-Stitching-05.czi
# ../npy/NoEpo_1-Stitching-05.0.crops.clas.npy.gz
# 0

for file in files:
    if file.endswith('FP.jpg'):
        img_file = file # ../1_ashley_label_img/1unitEpo_1-Stitching-01.0.crops.clas.npy.gz.jpg
        pattern = r"\.(\d{0,3})\.crops\.clas\.npy\.gz\.FP\.jpg"
        match = re.search(pattern, img_file)
        czi_file = re.sub(pattern, '.czi', img_file)
        npy_file = re.sub(r".FP.jpg$", '', img_file)
        log_file = img_file + '.txt'
        log = 'log_green/{}'.format(log_file)

        if match:
            I = match.groups()[0]
            img_path = os.path.join(img_dir, img_file)
            czi_path = os.path.join(czi_dir, czi_file)
            npy_path = os.path.join(npy_dir, npy_file)
            if file_exists_safe(img_path) and file_exists_safe(czi_path) and file_exists_safe(npy_path):
                print("python jgp2npy.green.py {} {} {} {} &> {}".format(img_path,czi_path, npy_path, I, log))
