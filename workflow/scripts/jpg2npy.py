import argparse
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # disables the warning for large Image
from ccount.blob.io import load_blobs, save_locs



'''
This script takes a user labeled {jpg} file, and an unlabeled blobs' <locs> file. 
Look for orange dots in positive blobs.
And outputs <label_locs> file in npy.gz format.

Steps: 
1. read <jpg> and <locs>, find all blobs' circles
2. look for an orange dot in each circle. the circles (blobs) with orange dots will be positives, all other blobs will be negatives
3. the positive/negative info will be saves in L column
4. label-locs [y,x,r,L] will be saved into a npy.gz file, internally it's an np.array
'''


def parse_args():
    parser = argparse.ArgumentParser(
        description='''
        This script takes a user labeled {jpg} file, and an unlabeled blobs' <locs> file. 
        Look for orange dots in positive blobs.
        And outputs <label_locs> file in npy.gz format.
        ''')
    parser.add_argument('-jpg', type=str, help="Path to the input JPG file (with orange dots in positive blob circles)")
    parser.add_argument('-locs', type=str, help="Path to the input blob locations in .npy.gz format, [y,x,r] expected, \
                        other columns will be ignored")
    parser.add_argument('-output', type=str,
                        help="The name of the output file, e.g. res/label_loc/plate.sceneIndex.npy.gz")

    args = parser.parse_args()
    print('''

        ''')
    print("cmd: ", args)
    return (args)


def read_colored_jpg(image_path):
    """
    Reads a JPG image and returns it as a colored PIL image object.

    Args:
        image_path: Path to the JPG image file.

    Returns:
        A PIL Image object representing the colored image.

    Raises:
        ValueError: If the image is not in JPG format or cannot be opened.
    """
    try:
        # Open the image in 'RGB' mode (ensures color)
        image = Image.open(image_path).convert('RGB')
        return image
    except OSError:
        raise ValueError(f"Error opening image file: {image_path}")


def main():
    # read scene img
    # get circles
    # find orange in circles
    # save results
    # czi = read_czi(czi_file)
    args = parse_args()
    img = read_colored_jpg(args.jpg)
    locs = load_blobs(args.locs)  # either crops or locs are converted to locs
    locs = locs



if __name__ == '__main__':
    main()
