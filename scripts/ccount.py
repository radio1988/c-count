from czifile import CziFile
from math import sqrt
from skimage import data, img_as_float
from skimage.draw import circle
from skimage import exposure
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage.transform import rescale, resize, downscale_local_mean
from IPython.display import clear_output
from random import randint
from time import sleep
import gzip

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os.path
import re
import time

# from ccount import *

def read_czi(fname):
    '''
    input: fname of czi file
    output: 2d numpy array
    '''
    with CziFile(fname) as czi:
        image_arrays = czi.asarray()
    image = image_arrays[0, 1, 0, 0, :, :, 0]  # real image
    print("{}: {}\n".format(fname, image.shape))
    return image


def equalize(image):
    '''
    input: image: 2d-array
    output: image: 2d-array, 0 black, 1 white
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    An algorithm for local contrast enhancement, that uses histograms computed over different tile regions of the image. Local details can therefore be enhanced even in regions that are darker or lighter than most of the image.
    '''
    return exposure.equalize_adapthist(image, clip_limit=0.03)


def down_scale(img, scaling_factor=2):
    '''
    input1: image
    input2: scaling_factor, # scale factor for each dim 1-> 1/1, 2 -> 1/2, 4 -> 1/4
    return: down scaled image array that is 1/scale-ratio in both x and y dimentions
    e.g.: block_small = down_scale(block, scaling_factor)  
    '''
    return resize(img, (img.shape[0] // scaling_factor, img.shape[1] // scaling_factor))


def find_blob(image_bright_blob_on_dark, scaling_factor = 2, 
    max_sigma=40, min_sigma=4, num_sigma=5, threshold=0.1, overlap=.0):
    '''
    input: gray scaled image with bright blob on dark background
    output: [n, 3] array of blob information, [y-locaiton, x-location, r-blob-radius]
    plot: original image of plates (dark colonies on light backgound)
    get squares of bright blobs

    Algorithm:
    Laplacian of Gaussian (LoG)
    This is the most accurate and slowest approach.
    It computes the Laplacian of Gaussian images with successively increasing standard deviation
    and stacks them up in a cube. Blobs are local maximas in this cube.
    Detecting larger blobs is especially slower because of larger kernel sizes during convolution.
    Only bright blobs on dark backgrounds are detected. See skimage.feature.blob_log for usage.
    '''
    print('image size', image_bright_blob_on_dark.shape)
    image_bright_blob_on_dark = down_scale(image_bright_blob_on_dark, scaling_factor)
    print('image-blob detection size', image_bright_blob_on_dark.shape)
    tic = time.time()
    blobs = blob_log(
        image_bright_blob_on_dark, 
        max_sigma=max_sigma, min_sigma=min_sigma, num_sigma=num_sigma, 
        threshold=threshold, overlap=overlap
        )
    blobs[:, 2] = blobs[:, 2] * sqrt(2)  # adjust r
    blobs = blobs * scaling_factor  # scale back coordinates
    toc = time.time()
    print("detection time: ", toc - tic)
    # larger num_sigma: more accurate boundry, slower
    # larger max_sigma: larger max blob size, slower
    # threshold: larger, less low contrast stuff
    # num_sigma=15 important for accuracy, step = 2.5
    # Compute radii in the 3rd column.
    return blobs


def vis_blob_on_block(blobs, block_img_equ, block_img_ori, 
    blob_extention_ratio=1.4, blob_extention_radius=2):
    '''
    blobs: blob info array [n, 3]
    block_img_equ: corresponding block_img equalized
    block_img_ori: block_img before equalization
    plot: plot block_img with blobs in yellow circles
    '''

    fig, axes = plt.subplots(2, 1, figsize=(60, 30), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].set_title('Blobs detected')
    ax[0].imshow(block_img_equ, 'gray', interpolation='nearest', clim=(0.0, 1.0))
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r * blob_extention_ratio + blob_extention_radius, color='yellow', linewidth=1,
                       fill=False)  # r*1.3 to get whole blob
        ax[0].add_patch(c)
    # ax[0].set_axis_off()

    ax[1].set_title("Input")
    ax[1].imshow(block_img_ori, 'gray', clim=(0.0, 1.0))

    plt.tight_layout()
    plt.show()


def hist_blobsize(blobs):
    '''
    show blob size distribution with histogram
    '''
    plt.title("Histogram of blob radius")
    plt.hist(blobs[:, 2], 40)
    plt.show()


def pad_with(vector, pad_width, iaxis, kwargs):
    '''
    to make np.pad in crop_blobs work
    '''
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def filter_blobs(blobs, r_min, r_max):
    '''
    filter blobs based on size of r
    '''
    flitered_blobs = blobs[blobs[:, 2] >= r_min,]
    flitered_blobs = flitered_blobs[flitered_blobs[:, 2] < r_max,]
    return flitered_blobs


def crop_blobs(blobs, block_img, block_row=-1, block_column=-1, crop_width=100,
               blob_extention_ratio=1.4, blob_extention_radius=2):
    '''
    input1: blobs, blob info [n, 3] [y, x, r]
    input2: block_img, corresponding block_img
    plt: cropped images
    return: cropped padded images in a flattened 2d-array, with meta data in the first 6 numbers
            - rows: blobs
            - columns: [y, x, r_, label, block_row, block_column, flattened cropped_blob_img (crop_width^2)]
                - y, x: corrd of centroid on original block_img
                - r_: extended radius
                - label: -1 = unlabeled; 1 = yes, 0 = no
                - block_row: block row num from whole image, start with 0, -1 means NA
                - block_column: same

    Algorithm:
    1. White padding
    2. Crop for each blob
    '''
    # White padding so that blobs on the edge can get cropped image
    padder = max(np.max(block_img), 1)
    padded = np.pad(block_img, crop_width, pad_with, padder=padder)  # 1 = white padding, 0 = black padding

    # crop for each blob
    flat_crops = np.empty((0, int(6 + 2 * crop_width * 2 * crop_width)))
    for blob in blobs:
        y, x, r = blob  # conter-intuitive order
        # print("the {}th blob  x:{} y:{} r:{}\n".format(i, x, y, r))
        y_ = int(y + crop_width)
        x_ = int(x + crop_width)  # adj for padding
        r_ = int(
            blob_extention_ratio * r + blob_extention_radius)  # extend circles, need to keep parameters sync between functions

        cropped_img = padded[y_ - crop_width: y_ + crop_width,
                      x_ - crop_width: x_ + crop_width]  # x coordinates use columns to locate, vise versa

        flat_crop = np.insert(cropped_img.flatten(), [0, 0, 0, 0, 0, 0],
                              [y, x, r_, -1, block_row, block_column])  # -1 unlabeled
        flat_crop = np.array([flat_crop])
        flat_crops = np.append(flat_crops, flat_crop, axis=0)
    return flat_crops


def plot_flat_crop(flat_crop, blob_extention_ratio=1, blob_extention_radius=0):
    '''
    input: one padded crop of a blob
    plt: yellow circle and hard-masked side-by-side
    columns: [y, x, r_, block_row, block_column, flattened cropped_blob_img (crop_width^2)]
    return: cropped image and hard-masked image
    '''
    # reshape
    [y, x, r, label, row, col] = flat_crop[0:6]
    r_ = r * blob_extention_ratio + blob_extention_radius

    flat = flat_crop[6:]
    w = int(sqrt(len(flat)) / 2)
    image = np.reshape(flat, (w + w, w + w))

    # hard mask creating training data
    mask = np.zeros((2 * w, 2 * w))  # zeros are masked to be black
    rr, cc = circle(w - 1, w - 1, min(r_, w - 1))
    mask[rr, cc] = 1  # 1 is white
    hard_masked = (1 - (1 - image) * mask)

    fig, axes = plt.subplots(1, 2, figsize=(8, 16), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].set_title('For human labeling\ncurrent label:{}\nradius:{}'.format(int(label), r_))
    ax[0].imshow(image, 'gray')
    c = plt.Circle((w - 1, w - 1), r_, color='yellow', linewidth=1, fill=False)
    ax[0].add_patch(c)
    ax[1].set_title('For model training\ncurrent label:{}\nradius:{}'.format(int(label), r_))
    ax[1].imshow(hard_masked, 'gray', clim=(0.0, 1.0))
    plt.tight_layout()
    plt.show()
    fig.canvas.draw()

    return image, hard_masked


def plot_flat_crops(flat_crops, blob_extention_ratio=1, blob_extention_radius=0):
    '''
    input: flat_crops
    task: plot padded crop and hard-masked crop side-by-side
    '''
    for flat_crop in flat_crops:
        plot_flat_crop(flat_crop, blob_extention_ratio=blob_extention_ratio, blob_extention_radius=blob_extention_radius)

# # Depreciated, use block_equalize + find_blob istread
# def split_image(image, 
#     block_height=2048, block_width=2048, 
#     crop_width=100,
#     blob_extention_ratio=1.4,
#     blob_extention_radius=2,
#     scaling_factor=2,
#     max_sigma=40, min_sigma=11, num_sigma=5, threshold=0.1, overlap=.0 
#     ):
#     '''
#     split
#     scale down
#     equalization
#     feed into find_blob
#     '''
#     height = block_height
#     width = block_width

#     image_flat_crops = np.empty((0, int(6 + 2 * crop_width * 2 * crop_width)))

#     r = 0
#     while (r + 1) * height <= image.shape[0]:
#         top = r * height
#         bottom = (r + 1) * height
#         c = 0
#         while (c + 1) * width <= image.shape[1]:
#             # get each block
#             left = c * width
#             right = (c + 1) * width
#             block = image[top:bottom, left:right]
#             print('block row:', r, '; column:', c, '; bottom-right pixcel:', bottom, right)
#             # scale down
#             block_small = down_scale(block, scaling_factor)  # scale factor for each dim 1-> 1/1, 2 -> 1/2, 4 -> 1/4
#             # equalization (good for blob detection, not nessessarily good for classifier) (todo)
#             block_small_equ = equalize(block_small)
#             # blob detection
#             blobs = find_blob(
#                 (1 - block_small_equ), 
#                 max_sigma=max_sigma, min_sigma=min_sigma, num_sigma=num_sigma, 
#                 threshold=threshold, overlap=overlap, 
#                 scaling_factor=scaling_factor
#             )  # reverse color, get bright blobs
#             # visualization of equalized block with blob yellow circles, and block before equalization
#             vis_blob_on_block(blobs, block_small_equ, block_small)
#             # create crop from blob (50x50 crops, with white padding)
#             block_flat_crops = crop_blobs(blobs, block_small, block_row=r, block_column=c, crop_width=crop_width) # image before
#             # equalization
#             image_flat_crops = np.append(image_flat_crops, block_flat_crops, axis=0)

#             print("{} block_flat_crops".format(len(block_flat_crops)))
#             print("{} image_flat_crops".format(len(image_flat_crops)))
#             # np.apply_along_axis( plot_flat_crop, axis=1, arr=block_flat_crops)
#             c += 1
#         r += 1

#     return image_flat_crops



def block_equalize(image, block_height=2048, block_width=2048):
    '''
    split
    equalization
    stitch and return
    '''
    image_equ = np.empty(image.shape)

    r = 0
    while (r + 1) * block_height <= image.shape[0]:
        top = r * block_height
        bottom = (r + 1) * block_height
        c = 0
        while (c + 1) * block_width <= image.shape[1]:
            # get each block
            left = c * block_width
            right = (c + 1) * block_width
            # equalization (good for blob detection, not nessessarily good for classifier) (todo)
            image_equ[top:bottom, left:right] = equalize(image[top:bottom, left:right]) # for each block
            # print('block row:', r, '; column:', c, '; bottom-right pixcel:', bottom, right)              
            c += 1
        r += 1

    return image_equ


def find_blobs_and_crop(image, image_equ, 
    crop_width=100,
    scaling_factor=2,
    max_sigma=40, min_sigma=11, num_sigma=5, threshold=0.1, overlap=.0,
    blob_extention_ratio=1.4, blob_extention_radius=2,
    ):
    '''
    '''
    image_flat_crops = np.empty((0, int(6 + 2 * crop_width * 2 * crop_width)))

    blobs = find_blob(
        (1 - image_equ), scaling_factor=scaling_factor,
        max_sigma=max_sigma, min_sigma=min_sigma, num_sigma=num_sigma, threshold=threshold, overlap=overlap
    )  # reverse color, get bright blobs

    vis_blob_on_block(blobs, image_equ, image, 
        blob_extention_ratio=blob_extention_ratio, blob_extention_radius=blob_extention_radius)

    # create crop from blob
    image_flat_crops = crop_blobs(blobs, image, crop_width=crop_width) # image before
    print("{} image_flat_crops".format(len(image_flat_crops)))

    return image_flat_crops




def pop_label_flat_crops(flat_crops, random=True, seed=1, skip_labeled=True):
    '''
    input: flat_crops
    task: plot padded crop and hard-masked crop side-by-side, and let user label them
    labels: -1 not-labeled, 0 NO, 1 YES
    output: labeled array in the original order
    '''
    print("Input: there are {} blobs unlabeled in {} blobs\n\n".format(sum(flat_crops[:, 3] == -1), len(flat_crops)))

    N = len(flat_crops)
    if random:
        np.random.seed(seed=seed)
        idx = np.random.permutation(N)
        np.random.seed()
    else:
        idx = np.arange(N)

    if skip_labeled:
        idx = idx[flat_crops[idx, 3] == -1]  # only keep unlabeled (-1)

    num_unlabeled = sum(flat_crops[:, 3] == -1)

    i = -1
    while i < len(idx):
        i += 1
        if i >= len(idx):
            break

        plot_flat_crop(flat_crops[idx[i], :])

        label = input('''labeling for the {}/{} unlabeled blob, 
yes=1, no=0, undistinguishable=u, skip=s, go-back=b, excape=e: '''.format(i + 1, num_unlabeled))

        if label == '1':
            flat_crops[idx[i], 3] = 1  # yes
        elif label == '0':
            flat_crops[idx[i], 3] = 0  # no
        elif label == 'u':
            flat_crops[idx[i], 3] = -2  # undistinguishable
        elif label == 's':
            pass
        elif label == 'b':
            i -= 2
        elif label == 'e':
            label = input('are you sure to quit?(y/n)')
            if label == 'y':
                print("there are {} blobs unlabeled\n\n".format(sum(flat_crops[:, 3] == -1)))
                print("labeling stopped manually")
                break
            else:
                print('continued')
                i -= 1
        else:
            print('invalid input, please try again')
            i -= 1

        print('new label: ', label, flat_crops[idx[i], 0:4])
        print("there are {} blobs unlabeled\n\n".format(sum(flat_crops[:, 3] == -1)))
        clear_output()

    return flat_crops


def load_blobs_db(in_db_name):
    '''
    use parameters: db_name
    input: db fname from user (xxx.npy)
    output: array (crops format)
    '''
    while 1:
        # in_db_name = in_db_name.strip()
        if os.path.isfile(in_db_name):
            if in_db_name.endswith('npy'):
                image_flat_crops = np.load(in_db_name)
            elif in_db_name.endswith('npy.gz'):
                f = gzip.GzipFile(in_db_name, "r")
                image_flat_crops = np.load(f)
            else:
                raise Exception ("db suffix not npy nor npy.gz")
                
            print("{} read into RAM".format(in_db_name))
            print("{} cropped blobs, {} pixcels in each blob".format(len(image_flat_crops),
                                                                     image_flat_crops.shape[1] - 6))
            print("{} unlabeled blobs".format(sum(image_flat_crops[:, 3] == -1)))
            break
        else:
            print("{} file not found".format(in_db_name))

    return image_flat_crops


def show_rand_crops(crops, label_filter="na", num_shown=5, 
    blob_extention_ratio=1, blob_extention_radius=0):
    '''
    blobs: the blobs crops
    label_filter: 0, 1, -1; "na" means no filter

    '''
    if (label_filter != 'na'):
        filtered_idx = crops[:, 3] == label_filter
        crops = crops[filtered_idx, :]

    if (len(crops) >= num_shown):
        randidx = np.random.choice(range(len(crops)), num_shown, replace=False)
        plot_flat_crops(crops[randidx, :], 
            blob_extention_ratio=blob_extention_ratio, blob_extention_radius=blob_extention_radius)
    elif (len(crops) > 0):
        plot_flat_crops(crops)
    else:
        print('num_blobs after filtering is 0')

# TEST SCALING and Equalization
# i = 0; j = 0; l = 2048
# block = image[2048*i : 2048*(i+1), 
#               2048*j : 2048*(j+1)]
# block_equ = equalize(block)
# block_equ_small = down_scale(block_equ, 2)

# print('block image: ', block.shape)
# print ("resized image: ", block_equ_small.shape)

# fig, axes = plt.subplots(1, 3, figsize=(20, 60), sharex=False, sharey=False)
# ax = axes.ravel()
# ax[0].set_title('input block')
# ax[0].imshow(block, 'gray')
# ax[1].set_title('equalized block')
# ax[1].imshow(block_equ, 'gray')
# ax[2].set_title('scaled block')
# ax[2].imshow(block_equ_small, 'gray')