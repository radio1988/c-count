import numpy as np
import math, os, re, sys, cv2, time
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None   # disables the warning
from ccount.blob.io import load_locs, save_locs
from ccount.img.read_czi import read_czi, parse_image_obj
from ccount.blob.plot import visualize_blob_detection, visualize_blob_compare
from ccount.blob.misc import crops_stat

"""
usage: python jpg2npy.py <labeled.jpg> <image.czi> <all_locs.npy.gz> <index> <outname>

Args:
- labeled.jpg (str): images with user labeled orange dots in the circles
- image.czi (str): raw image with four scenes 
- all_loc.npy.gz: loc file with y,x,r saved or y,x,r,L saved. all locs will be scanned for label dots inside the circle
- index: scene index [0, 1, 2, 3]
- outname: prefix for outputs, e.g. sample1

Process:
- Read czi, get scene for the index
- Find circles with y,x,r from locs file
- Look for color inside the cicles (has orange dot or not, one pixel is good enough)
- if orange, mark pos, else, no
- save results in locs file (y,x,r,L)

Params:
- Orange:
        red > 200 and green > 70 and blue < 20 and red > green * 1.4
        # ashley: color_inside: (210, 86, 16)
        # logan: color_inside: (243, 151, 14)
        # john: color_inside: (248, 151, 12)


Returns:
- locs file (y,x,r,L)

Example:
under ccount/workflow/scripts/test/jpg2npy/
python jpg2npy.py ashley.jpg 1U_Epo_1-Stitching-01.czi 1U_Epo_1-Stitching-01.1.crops.clas.npy.gz 1 ashley
"""

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


def is_orange(rgb_color):
    """
    Checks if an RGB color is considered orange based on a predefined range.

    Args:
        rgb_color: A tuple representing an RGB color (red, green, blue).

    Returns:
        True if the color falls within the orange range, False otherwise.
    """
    # Define orange color range (adjust as needed)
    red, green, blue = rgb_color
    if red > 200 and green > 70 and blue < 20 and red > green * 1.4:
        # ashley: color_inside: (210, 86, 16)
        # logan: color_inside: (243, 151, 14)
        # john: color_inside: (248, 151, 12)
        return True
    else:
        return False


def find_unique_colors_in_circle(image, x, y, r):
    """
    Finds all unique colors within a circle in a PIL image.

    Args:
        image: A PIL Image object representing the image.
        x: X-coordinate of the circle's center.
        y: Y-coordinate of the circle's center.
        r: Radius of the circle.

    Returns:
        A set containing all unique color tuples (R, G, B) found within the circle.
    """
    unique_colors = set()
    width, height = image.size
    # Iterate through pixels within the circle's bounding box
    for i in range(max(0, x - r), min(width - 1, x + r) + 1):
        for j in range(max(0, y - r), min(height - 1, y + r) + 1):
            # Check if the pixel lies within the circle
            if ((i - x) ** 2 + (j - y) ** 2) <= r ** 2:
                # Get the pixel's RGB color
                pixel_color = image.getpixel((i, j))
                unique_colors.add(pixel_color)
    return unique_colors


def find_unique_colors_at_circle(image, x, y, r):
    """
    Finds all unique colors at the circle in a PIL image.
    circle is counted as r-5 to r in radius

    Args:
        image: A PIL Image object representing the image.
        x: X-coordinate of the circle's center.
        y: Y-coordinate of the circle's center.
        r: Radius of the circle.

    Returns:
        A set containing all unique color tuples (R, G, B) found within the circle.
    """
    unique_colors = set()
    width, height = image.size  # test, reversed?
    # Iterate through pixels within the circle's bounding box
    for i in range(max(0, x - r), min(width - 1, x + r) + 1):
        for j in range(max(0, y - r), min(height - 1, y + r) + 1):
            # Check if the pixel lies within the circle
            if ((i - x) ** 2 + (j - y) ** 2) <= r ** 2 and ((i - x) ** 2 + (j - y) ** 2) >= (r - 5) ** 2:
                # Get the pixel's RGB color
                pixel_color = image.getpixel((i, j))
                unique_colors.add(pixel_color)
    return unique_colors


def crop_square_around_center(image, x, y, square_size=200):
    """
    Crops a square around a center point in a PIL image.
    For testing

    Args:
        image: A PIL Image object representing the image.
        x: X-coordinate of the center point.
        y: Y-coordinate of the center point.
        square_size: Size of the desired square crop (default: 200).

    Returns:
        A new PIL Image object representing the cropped square.

    Raises:
        ValueError: If the requested crop goes outside the image boundaries.
    """
    width, height = image.size
    half_square = square_size // 2
    # Ensure crop stays within image boundaries
    left = max(0, x - half_square)
    top = max(0, y - half_square)
    right = min(width, x + half_square)
    bottom = min(height, y + half_square)
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image  # Return cropped image and adjusted center


def find_non_white_pixels(fname):
    """
    Finds non-white ( < 254) image within the input image
    So you can remove white backgrounds, and only keep the foreground image
    :param fname:
    """
    image = cv2.imread(fname)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = 254  # Experiment with this value
    thresholded_image = np.where(gray_image < thresh, 255, 0)  # gray as 255, white as zero
    nonzero_pixels = np.nonzero(thresholded_image)  # non-zero indices
    top = np.min(nonzero_pixels[0]) + 40  # todo: current setting specific, hard code
    left = np.min(nonzero_pixels[1]) + 50
    bottom = np.max(nonzero_pixels[0]) - 30
    right = np.max(nonzero_pixels[1])
    print(f"Bounding box coordinates (left, top, right, bottom): ({left}, {top}, {right}, {bottom})")
    return (left, top, right, bottom)


def find_non_white_boundaries(fname, min_foreground_density=0.2):
    """
    Input: filename
    Output: the boundaries coordinates on the original image, for non-white image on white canvas
    Params: have to have less than {max_white_density} white pixels in the row/column
    """
    image = cv2.imread(fname)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = 254  # Experiment with this value
    thresholded_image = np.where(gray_image < thresh, 255, 0)  # gray as 255, white as zero
    height, width = thresholded_image.shape
    for r, row in enumerate(thresholded_image):
        black_count = np.count_nonzero(row)
        if black_count / width >= min_foreground_density:
            top = r
            break
    for r in reversed(range(height)):
        row = thresholded_image[r, :]
        black_count = np.count_nonzero(row)
        if black_count / width >= min_foreground_density:
            bottom = r
            break
    for c in range(width):
        column = thresholded_image[:, c]
        black_count = np.count_nonzero(column)
        if black_count / height >= min_foreground_density:
            left = c
            break
    for c in reversed(range(width)):
        column = thresholded_image[:, c]
        black_count = np.count_nonzero(column)
        if black_count / height >= min_foreground_density:
            right = c
            break
    return (left, top, right, bottom)


def find_dominant_color(colors):
    """
    Reads RGB colors
    Output the color with most pixels, if the color is 2 times brighter than other channels
    For RGB only
    :param colors: ((235, 156, 151),(237, 136, 152))
    :return:
        max_key: 0,1, or 2 (R,G,B)
        C: count for each color
    """
    n_pixel = 0
    C = {}
    C[0] = 0
    C[1] = 0
    C[2] = 0
    for color in colors:
        n_pixel += 1
        max_i = color.index(max(color))  # most bright color
        other_is = [num for num in [0, 1, 2] if num != max_i]
        other_colors = [color[i] for i in other_is]
        if color[max_i] / (np.mean(other_colors) + 0.0001) > 2:
            C[max_i] += 1
    max_key = max(C, key=C.get)
    # If all gray
    if sum(C.values()) < 1:
        max_key = None
    if len(set(C.values())) < 2:
        max_key = None
    return max_key, C


# arguments
test = False

if len(sys.argv) != 6:
    raise Exception ("command error")
else:
    img_file = sys.argv[1]  # jpg-labeled, e.g.  "1unitEpo_1-Stitching-01.czi"
    czi_file = sys.argv[2]  # czi (raw img), e.g. '1unitEpo_1-Stitching-01.1.crops.clas.npy.gz.jpg'
    npy_file = sys.argv[3]  # locs or crops file (locs corresponding to circles in jpg-labeled), e.g. '../npy/1unitEpo_1-Stitching-01.1.crops.clas.npy.gz'
    I = sys.argv[4]  # [0,1,2,3] the index for czi files
    outname = sys.argv[5]  # for outname, e.g. '1unitEpo_1-Stitching-01.1.npy.gz'
    if not outname.endswith('npy.gz'):
        raise Exception ("outname has to end with npy.gz")

# read czi
czi = read_czi(czi_file)
czi_img = parse_image_obj(czi, I)  # czi_img.shape (8635, 10620) (h,w); 10620/8635 1.229878401852924

# read locs
npy_based_locs = load_locs(npy_file)  # either crops or locs are converted to locs

if npy_based_locs.shape[1] > 3:
    npy_based_locs = npy_based_locs[:, 0:4]  # y,x,r,L
else:
    npy_based_locs = npy_based_locs[:, 0:3]  # y,x,r

dot_based_locs = npy_based_locs.copy()
dot_based_locs = dot_based_locs[:, 0:3]  # remove labels
zeros_col = np.zeros((npy_based_locs.shape[0], 1))
dot_based_locs = np.concatenate((dot_based_locs, zeros_col), axis=1)  # new labels of all negatives
# circle_based_locs = dot_based_locs.copy()

img = read_colored_jpg(img_file)  # img refers to manually labeled image
left, top, right, bottom = find_non_white_boundaries(img_file)  # Find foreground location
img = img.crop((left, top, right, bottom))
img_gray = np.array(img.convert("L"))
# cv2.imwrite('test.jpg', img_gray)  # test coordinates
scale = math.sqrt(img.size[0] ** 2 + img.size[1] ** 2) / math.sqrt(
    czi_img.shape[1] ** 2 + czi_img.shape[0] ** 2)  # based on diagonal length


# process blobs in npy_based_locs one by one
D = {0: 1, 1: 0, 2: 0, None: None}  # label conversion dictionary,  0,1,2 (R,G,B) to 0,1 (Labels)
for i in range(len(npy_based_locs)):
    y = int(npy_based_locs[i][0] * scale)  # max y: 8628
    x = int(npy_based_locs[i][1] * scale)  # max x: 10616
    r = int((npy_based_locs[i][2] * 1.4 + 10) * scale + 1)  # expanded as usual x1.4 and +10

    # look at color OF the circle
    colors = find_unique_colors_at_circle(img, x, y, r)  # find the color of the circle (r to r-5)
    max_color, C = find_dominant_color(colors)
    circle_label = D[max_color]  # label read from circle in jpg, 0, 1, None


    # look at the color of the dot IN the circle
    colors_inside = find_unique_colors_in_circle(img, x, y, r)
    dot_label = 0
    for color_inside in colors_inside:
        if is_orange(color_inside):  # any single pixel is organge (re-labeling)
            dot_label = 1
            dot_based_locs[i][3] = 1  # mod label
            break
    if test and dot_label > 0:
        print("dot_label:", dot_label)
        if dot_label > 0:
            print('color_inside:', color_inside)
        print("circle_label:", circle_label)
        if npy_based_locs.shape[1] > 3:
            npy_label = npy_based_locs[i][3]
            print("\nnpy_label:", npy_label)

stats_dot = crops_stat(dot_based_locs)
print("dot based stats:", stats_dot)
#stats_cir = crops_stat(circle_based_locs)
#print("circle based stats:", stats_cir)

if npy_based_locs.shape[1] > 3:
    stats_npy = crops_stat(npy_based_locs)
    print("npy based stats:", stats_npy)


outdir = os.path.dirname(outname)
os.makedirs(outdir, exist_ok=True)
save_locs(dot_based_locs,  outname)
visualize_blob_detection(czi_img, dot_based_locs, fname=outname.replace(".npy.gz",".jpg"))
