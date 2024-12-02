import argparse
import os, re, sys
import math
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # disables the warning for large Image
from ccount_utils.blob import load_blobs, save_locs, crop_blobs, get_blob_statistics
from ccount_utils.img import read_czi, parse_image_obj
from ccount_utils.blob import visualize_blobs_on_img, visualize_blob_compare

blob_extention_ratio = 1.4  # todo: load yaml
blob_extention_radius = 10

'''
This script takes a user labeled {jpg} file, and an unlabeled blobs' <locs> file. 
Look for orange dots in positive blobs.
And outputs <label_locs> file in npy.gz format.

It needs czi file and sceneIndex to scale jpg back to czi yxr coordinates

Process: 
- Read czi, get scene for the index
- Read jpg, scale back coordinates with czi.scene
- Find circles with y,x,r from locs file, in the jpg-img
- Look for color inside the circles (has orange dot or not, one pixel is good enough)
- if orange pixels detected (>9), mark positive, else, negative, in L column
- save results in locs file (y,x,r,L)
'''


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
        '-locs', type=str, required=True,
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
    print("\nCommand typed: ", " ".join(sys.argv))
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
        blob_extention_ratio=1.4, blob_extention_radius=10
):
    """
    If >9 orange pixels are found in the corresponding circle, mark positive, otherwise negative
    @param jpg_img:
    @param jpg_locs:
    @return: jpg_labeled_locs [yxrL] with labels
    """
    labels = []
    for i in range(len(jpg_locs)):
        y = int(jpg_locs[i][0])  # max y: 8628
        x = int(jpg_locs[i][1])  # max x: 10616
        r = int((jpg_locs[i][2] * blob_extention_ratio + blob_extention_radius) + 1)  # expanded as usual x1.4 and +10

        # look at color OF the circle
        # colors_at_circle = find_unique_colors_at_circle(img, x, y, r)  # find the color of the circle (r to r-5)
        # max_color, C = find_dominant_color(colors_at_circle)
        # circle_label = D[max_color]  # label read from circle in jpg, 0, 1, None

        # look at the color of the dot IN the circle
        count = count_orange_pixels_in_circle(jpg_img, x, y, r)
        # colors_inside = find_unique_colors_in_circle(jpg_img, x, y, r)

        if count > 9:
            labels.append(int(1))
        else:
            labels.append(int(0))

    return labels


def count_orange_pixels_in_circle(image, x, y, r):
    """
    Count num of orange pixels within a circle in a PIL image.

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
    # Iterate through pixels within the circle's bounding box
    for i in range(max(0, x - r), min(width - 1, x + r) + 1):
        for j in range(max(0, y - r), min(height - 1, y + r) + 1):
            # Check if the pixel lies within the circle
            if ((i - x) ** 2 + (j - y) ** 2) <= r ** 2:
                # Get the pixel's RGB color
                pixel_color = image.getpixel((i, j))
                if is_pixel_orange(pixel_color):
                    count += 1
    # print('count: {}, x: {}, y: {}, r{}'.format(count, x, y, r))
    return count


# def find_unique_colors_at_circle(image, x, y, r):
#     """
#     Finds all unique colors at the circle in a PIL image.
#     circle is counted as r-5 to r in radius
#
#     Args:
#         image: A PIL Image object representing the image.
#         x: X-coordinate of the circle's center.
#         y: Y-coordinate of the circle's center.
#         r: Radius of the circle.
#
#     Returns:
#         A set containing all unique color tuples (R, G, B) found within the circle.
#     """
#     unique_colors = set()
#     width, height = image.size  # test, reversed?
#     # Iterate through pixels within the circle's bounding box
#     for i in range(max(0, x - r), min(width - 1, x + r) + 1):
#         for j in range(max(0, y - r), min(height - 1, y + r) + 1):
#             # Check if the pixel lies within the circle
#             if ((i - x) ** 2 + (j - y) ** 2) <= r ** 2 and ((i - x) ** 2 + (j - y) ** 2) >= (r - 5) ** 2:
#                 # Get the pixel's RGB color
#                 pixel_color = image.getpixel((i, j))
#                 unique_colors.add(pixel_color)
#     return unique_colors
#
#
# def find_dominant_color(colors):
#     """
#     Reads RGB colors
#     Output the color with most pixels, if the color is 2 times brighter than other channels
#     For RGB only
#     :param colors: ((235, 156, 151),(237, 136, 152))
#     :return:
#         max_key: 0,1, or 2 (R,G,B)
#         C: count for each color
#     """
#     n_pixel = 0
#     C = {}
#     C[0] = 0
#     C[1] = 0
#     C[2] = 0
#     for color in colors:
#         n_pixel += 1
#         max_i = color.index(max(color))  # most bright color
#         other_is = [num for num in [0, 1, 2] if num != max_i]
#         other_colors = [color[i] for i in other_is]
#         if color[max_i] / (np.mean(other_colors) + 0.0001) > 2:
#             C[max_i] += 1
#     max_key = max(C, key=C.get)
#     # If all gray
#     if sum(C.values()) < 1:
#         max_key = None
#     if len(set(C.values())) < 2:
#         max_key = None
#     return max_key, C

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
    if red > 200 and 160 > green > 70 and blue < 20 and red > green * 1.4:
        # ashley: color_inside: (210, 86, 16)
        # logan: color_inside: (243, 151, 14)
        # john: color_inside: (248, 151, 12)
        return True
    else:
        return False


def find_orange_dot_in_blob(crop):
    """
    Input a single crop and see if it has orange dot in it
    @param crop:
    @return:
    """
    return True


def main():
    args = parse_args()

    czi_obj = read_czi(args.czi)
    czi_img = parse_image_obj(czi_obj,
                              args.sceneIndex)  # czi_img.shape (8635, 10620) (h,w); 10620/8635 1.229878401852924

    czi_locs = load_blobs(args.locs)  # czi coordinates are larger than jpg_locs
    if czi_locs.shape[1] > 3:
        czi_locs = czi_locs[:, 0:4]

    jpg_img = read_colored_jpg(args.jpg)
    print('input jpg size', jpg_img.size)
    jpg_img = remove_jpg_white_canvas(jpg_img, args.jpg)
    print('jpg size after removing white canvas', jpg_img.size)

    scale = math.sqrt(jpg_img.size[0] ** 2 + jpg_img.size[1] ** 2) / \
            math.sqrt(czi_img.shape[1] ** 2 + czi_img.shape[0] ** 2)  # based on diagonal length
    print("scale from czi to jpg is {}".format(round(scale, 3)))
    jpg_locs = czi_locs * scale
    print("example czi locs:\n", czi_locs[0:3, ])
    print("example jpg locs:\n", jpg_locs[0:3, ], "\n")

    jpg_labels = get_labels(jpg_img, jpg_locs, blob_extention_ratio, blob_extention_radius)
    jpg_labels = np.array(jpg_labels).reshape(-1, 1)
    czi_locs_labeled = np.hstack((czi_locs, jpg_labels))
    print("example labeled jpg locs:\n", czi_locs_labeled[0:3, ])
    get_blob_statistics(czi_locs_labeled)

    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_locs(czi_locs_labeled, args.output)

    visualize_blobs_on_img(czi_img, czi_locs_labeled, fname=args.output.replace(".npy.gz", ".jpg"))


if __name__ == '__main__':
    main()
