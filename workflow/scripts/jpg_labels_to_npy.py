"""
This script takes a user labeled {jpg} file, and an unlabeled blobs' <locs> file.
Look for orange dots in positive blobs.
And outputs <label_locs> file in npy.gz format.

It needs czi file and sceneIndex to scale jpg back to czi yxr coordinates

Process:
- Read czi, get scene for the index
- Read jpg, scale back coordinates with czi.scene
- Find circles with y,x,r from locs file, in the jpg-img
- Look for orange pixels inside the circles
- Form orange-marks with connected pixels
- The size of the largest orange-mark >15, the blob is positive, else, negative, in L column
- save results in locs file (y,x,r,L)
"""

import argparse
import math
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import label

Image.MAX_IMAGE_PIXELS = None  # disables the warning for large Image
from ccount_utils.blob import load_blobs, save_locs, get_blob_statistics
from ccount_utils.img import read_czi, parse_image_obj
from ccount_utils.blob import visualize_blobs_on_img

blob_extention_ratio = 1.4  # todo: load yaml
blob_extention_radius = 10


# todo: unlabeled blob visualization based jpg only have one line of title, \
#  labeled has two, can cause alignment problems


def validate_npy_gz(file_path):
    """Validate that the file ends with .npy.gz."""
    if not file_path.endswith(".npy.gz"):
        raise argparse.ArgumentTypeError(f"The file '{file_path}' must end with .npy.gz")
    if not os.path.isfile(file_path):
        raise argparse.ArgumentTypeError(f"The file '{file_path}' does not exist")
    return file_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='''
        This script takes a user labeled {jpg} file, and an unlabeled blobs' <locs> file. 
        Look for orange dots in positive blobs.
        And outputs <label_locs> file in npy.gz format.
        ''')
    parser.add_argument(
        '-czi', type=str, required=True,
        help="Path to the czi raw image, need this to scale jpg image to locs yxr \
        (jpg size sometimes different)"
    )
    parser.add_argument(
        '-jpg', type=str, required=True,
        help="Path to the input JPG file (with orange dots in positive blob circles)"
    )
    parser.add_argument(
        '-locs', type=validate_npy_gz, required=True,
        help="Path to the input blob locations in .npy.gz format, \
        [y,x,r] expected, other columns will be ignored"
    )
    parser.add_argument(
        '-sceneIndex', type=str, required=True,
        help="sceneIndex of the jpg in czi, also help scale yxr coordinates"
    )
    parser.add_argument(
        '-output', type=str, required=True,
        help="The name of the output file, \
        e.g. res/label_loc/plate.sceneIndex.npy.gz"
    )
    args = parser.parse_args()
    print("\nCommand typed: ", " ".join(sys.argv), "\n")
    return args


def read_colored_jpg(image_path):
    """
    Reads a JPG image and returns it as a colored PIL image object.
    """
    print("\n<read_colored_jpg>")
    print("reading: ", image_path)
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except OSError:
        raise ValueError(f"Error opening image file: {image_path}")


def find_non_white_boundaries(fname, min_foreground_density=0.2):
    """
    Input: filename
    Output: the boundaries coordinates on the original image, for non-white image on white canvas
    Params: have to have less than {max_white_density} white pixels in the row/column
    """
    print("<find_non_white_boundaries> from jpg")
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


def remove_jpg_white_canvas(jpg_img, jpg_fname):
    left, top, right, bottom = find_non_white_boundaries(jpg_fname)  # Find foreground location
    jpg_img = jpg_img.crop((left, top, right, bottom))
    return jpg_img


def get_labels(
        jpg_img, jpg_locs,
        blob_extention_ratio=1.4, blob_extention_radius=10,
        TEST=False
):
    """
    If >15 connected orange pixels are found in the corresponding circle, mark positive, otherwise negative
    @param jpg_img:
    @param jpg_locs:
    @return: jpg_labeled_locs [yxrL] with labels
    """
    labels = []
    blob_orange_pixel_counts = []
    for i in range(len(jpg_locs)):  # for each blob in the jpg
        y = int(jpg_locs[i][0])  # max y: 8628
        x = int(jpg_locs[i][1])  # max x: 10616
        r = int((jpg_locs[i][2] * blob_extention_ratio + blob_extention_radius) + 1)  # expanded as usual x1.4 and +10

        # look at color OF the circle
        # colors_at_circle = find_unique_colors_at_circle(img, x, y, r)  # find the color of the circle (r to r-5)
        # max_color, C = find_dominant_color(colors_at_circle)
        # circle_label = D[max_color]  # label read from circle in jpg, 0, 1, None

        # look at the color of the dot IN the circle
        count = count_orange_pixels_in_circle(jpg_img, x, y, r)
        blob_orange_pixel_counts.append(count)
        # colors_inside = find_unique_colors_in_circle(jpg_img, x, y, r)

        if count > 15:
            labels.append(int(1))
        else:
            labels.append(int(0))

    if TEST:
        plt.hist(blob_orange_pixel_counts, bins=40)
        plt.yscale('log')  # Set y-axis to logarithmic scale
        plt.title("Histogram of Counts for orange Pixels")
        plt.savefig("hist.connected_orange_pixel_count.pdf", dpi=300)

    return labels


def count_orange_pixels_in_circle(image, x, y, r):
    """
    Count the number of orange pixels within a circular region in a PIL image.
    Only connected orange pixels are considered, forming “orange-marks”
    The function identifies and counts pixels in the largest connected blob of orange pixels within the circle \
    and reports the size of this largest orange-marks.

    Args:
        image: A PIL Image object representing the image.
        x: X-coordinate of the circle's center.
        y: Y-coordinate of the circle's center.
        r: Radius of the circle.

    Returns:
        A set containing all unique color tuples (R, G, B) found within the circle.
    """
    count = 0
    width, height = image.size
    orange_locations = set()

    box_min_x = max(0, x - r)
    box_max_x = min(width - 1, x + r) + 1
    box_min_y = max(0, y - r)
    box_max_y = min(height - 1, y + r) + 1
    for i in range(box_min_x, box_max_x):  # bounding box
        for j in range(box_min_y, box_max_y):
            if ((i - x) ** 2 + (j - y) ** 2) <= r ** 2:
                pixel_color = image.getpixel((i, j))
                if is_pixel_orange(pixel_color):
                    count += 1
                    orange_locations.add((i, j))

    mask = np.zeros((box_max_x - box_min_x, box_max_y - box_min_y), dtype=int)
    for x, y in orange_locations:
        mask[x - box_min_x, y - box_min_y] = 1

    max_cluster_pixel_count = find_largest_cluster(mask)
    # print('count: {}, x: {}, y: {}, r{}'.format(count, x, y, r))
    return max_cluster_pixel_count


def find_largest_cluster(mask):
    """
    Finds clusters in a binary mask and returns the pixel count of the largest cluster.

    Args:
        mask (numpy.ndarray): Binary mask (2D array) where clusters are represented by 1s.

    Returns:
        int: Pixel count of the largest cluster.
    """
    # Label connected components
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled_array, num_features = label(mask, structure=structure)

    if num_features == 0:
        return 0  # No clusters found

    # Calculate the size of each cluster
    cluster_sizes = np.bincount(labeled_array.ravel())[1:]  # Exclude background (label 0)

    # Return the size of the largest cluster
    return cluster_sizes.max()


def is_pixel_orange(rgb_color):
    """
    Checks if an RGB color is considered orange based on a predefined range.

    Args:
        rgb_color: A tuple representing an RGB color (red, green, blue).

    Returns:
        True if the color falls within the orange range, False otherwise.
    """
    # Define orange color range (adjust as needed)
    red, green, blue = rgb_color
    if red > 200 and 160 > green > 70 and blue < 70 and red > green * 1.4:
        # ashley: color_inside: (210, 86, 16)
        # logan: color_inside: (243, 151, 14)
        # john: color_inside: (248, 151, 12)
        # mac preview orange (255, ~140, 0)
        # ashley FL-A 240, 154, 50
        return True
    else:
        return False


def main():
    args = parse_args()
    TEST = False

    czi_obj = read_czi(args.czi)
    czi_img = parse_image_obj(
        czi_obj,
        args.sceneIndex)  # czi_img.shape (8635, 10620) (h,w); w/h=1.22

    czi_locs = load_blobs(args.locs)  # czi coordinates are larger than jpg_locs
    if czi_locs.shape[1] > 3:
        czi_locs = czi_locs[:, 0:3]  # yxr only, no L needed

    jpg_img = read_colored_jpg(args.jpg)
    print('input jpg size', jpg_img.size)
    jpg_img = remove_jpg_white_canvas(jpg_img, args.jpg)
    print('jpg size after removing white canvas', jpg_img.size)

    print("\nexample czi sized locs:\n", czi_locs[0:4, :])
    scale = math.sqrt(jpg_img.size[0] ** 2 + jpg_img.size[1] ** 2) / \
            math.sqrt(czi_img.shape[1] ** 2 + czi_img.shape[0] ** 2)  # based on diagonal length
    print("scale from czi to jpg is {}".format(round(scale, 3)))
    jpg_locs = czi_locs.copy()
    jpg_locs[:, 0:3] = czi_locs[:, 0:3] * scale
    print("example jpg sized locs(should be scaled):\n", jpg_locs[0:4, :])

    jpg_labels = get_labels(jpg_img, jpg_locs, blob_extention_ratio, blob_extention_radius, TEST)
    jpg_labels = np.array(jpg_labels).reshape(-1, 1)
    czi_locs_labeled = np.hstack((czi_locs, jpg_labels))
    print("example output locs(should be czi sized):\n", czi_locs_labeled[0:4, ])
    get_blob_statistics(czi_locs_labeled)

    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_locs(czi_locs_labeled, args.output)

    visualize_blobs_on_img(czi_img, czi_locs_labeled, fname=args.output.replace(".npy.gz", ".jpg"))


if __name__ == '__main__':
    main()
