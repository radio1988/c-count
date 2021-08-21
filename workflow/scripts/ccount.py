import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path
import re
import time
import tracemalloc
import gc



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


def read_czi(fname, Format="2018", concatenation=False):
    '''
    input: fname of czi file
    output: 2d numpy array
    assuming input czi format (n, 1, :, :, 1)
    e.g. (4, 1, 70759, 65864, 1)

    '''
    from czifile import CziFile

    if fname.endswith('czi'):
        with CziFile(fname) as czi:
            image_arrays = czi.asarray()  # 129s, Current memory usage is 735.235163MB; Peak was 40143.710599MB
            print(image_arrays.shape)
    elif fname.endswith('czi.gz'):
        raise Exception("todo")
        # with gzip.open(fname, 'rb') as f:
        #     with CziFile(f) as czi:
        #         image_arrays = czi.asarray()
    else:
        raise Exception("input czi/czi.gz file type error\n")
        
    Format = Format.strip()

    if Format == "2018":
        # reading (need 38 GB RAM) todo: use int16 if possible
        image = image_arrays[0, 1, 0, 0, :, :, 0]  # real image
        print("{}: {}\n".format(fname, image.shape))
        return image 
    elif Format == "2019":        
        # Find Box with info: todo faster by https://kite.com/python/docs/PIL.Image.Image.getbbox  
        # todo: int to float slow and large RAM usage, must change, use int16 if possible, done?
        lst = []
        for i in range(0,image_arrays.shape[0]): # loop iter does not change process time 
            print("For ", i, " area in czi")
            image = image_arrays[i, 0, :,  :, 0] # 0s
            nz_image = np.nonzero(image)  # process_time(),36s, most time taken here, 1.4GB RAM with tracemalloc
            nz0 = np.unique(nz_image[0]) # 1.5s
            nz1 = np.unique(nz_image[1]) # 2.4s
            del nz_image
            n = gc.collect()
            print("Number of unreachable objects collected by GC:", n)  
            if len(nz0) < 2 or len(nz1) < 2: 
                continue
            print(nz0, nz0.shape)
            print(nz1, nz1.shape)
            image = image[min(nz0):max(nz0), min(nz1):max(nz1)]  # 0s
            lst.append(image) # 0s
        
        if concatenation:
            # padding
            heights = [x.shape[0] for x in lst]
            widths = [x.shape[1] for x in lst]
            max(heights)
            max(widths)
            for (i,image) in enumerate(lst):
                print(image.shape, i)
                pad_h = max(heights) - image.shape[0]
                pad_w = max(widths) - image.shape[1]
                lst[i] = np.pad(image, [[0,pad_h],[0,pad_w]], "constant")
                
            # concat: use a long wide image instead to adjust for unknown number of scanns
            image = np.hstack(lst)
            print("shape of whole picture {}: {}\n".format(fname, image.shape))
            return image
        else:
            # return a list of single are images
            return lst #[image0, image1, image2 ..]
    else:
        raise Exception("image format error\n")
        return None


def equalize(image):
    '''
    input: image: 2d-array
    output: image: 2d-array, 0 black, 1 white
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    An algorithm for local contrast enhancement, that uses histograms computed over different tile regions of the image. Local details can therefore be enhanced even in regions that are darker or lighter than most of the image.
    '''
    import warnings
    warnings.filterwarnings("ignore")
    #print("Equalizing img of size:", image.shape)
    return exposure.equalize_adapthist(image, clip_limit=0.01)  # Aug, 2019, cleaner image than 0.03


def down_scale(img, scaling_factor=2):
    '''
    input1: image
    input2: scaling_factor, # scale factor for each dim 1-> 1/1, 2 -> 1/2, 4 -> 1/4
    return: down scaled image array that is 1/scale-ratio in both x and y dimentions
    e.g.: block_small = down_scale(block, scaling_factor)  
    '''
    return resize(img, (img.shape[0] // scaling_factor, img.shape[1] // scaling_factor))


def blobs_stat(blobs):
    '''
    print summary of labels in blobs
    :param blobs:
    :return:
    '''
    print("{} Yes, {} No, {} Uncertain, {} Unlabeled".format(
        sum(blobs[:, 3] == 1),
        sum(blobs[:, 3] == 0),
        sum(blobs[:, 3] == -2),
        sum(blobs[:, 3] == -1),
    ))


def find_blob(image_bright_blob_on_dark, scaling_factor = 2, 
    max_sigma=40, min_sigma=4, num_sigma=5, threshold=0.1, overlap=.0):
    '''
    input: gray scaled image with bright blob on dark background
    output: [n, 3] array of blob information, [y-locaiton, x-location, r-blob-radius] !!!
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
    blob_extention_ratio=1.4, blob_extention_radius=2, scaling = 8, fname=None):
    '''
    blobs: blob info array [n, 0:3]
    block_img_equ: corresponding block_img equalized
    block_img_ori: block_img before equalization
    plot: plot block_img with blobs in yellow circles
    '''
    print('scaling of visualization is ', scaling)
    blobs = blobs[:, 0:3]
    blobs = blobs/scaling
    block_img_equ = down_scale(block_img_equ, scaling)
    block_img_ori = down_scale(block_img_ori, scaling)
    

    fig, axes = plt.subplots(2, 1, figsize=(40, 20), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].set_title('Equalized Image')
    ax[0].imshow(block_img_equ, 'gray', interpolation='nearest', clim=(0.0, 1.0))
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), 
                       r * blob_extention_ratio + blob_extention_radius, 
                       color=(0.9, 0.9, 0, 0.5), linewidth=1,
                       fill=False)  # r*1.3 to get whole blob
        ax[0].add_patch(c)
    # ax[0].set_axis_off()
    ax[1].set_title("Original Image")
    ax[1].imshow(block_img_ori, 'gray', interpolation='nearest', clim=(0.0, 1.0))
    for blob in blobs:
        y, x, r = blob
        d = plt.Circle((x, y), 
                       r * blob_extention_ratio + blob_extention_radius, 
                       color=(0.9, 0.9, 0, 0.5), linewidth=1,
                       fill=False)  # r*1.3 to get whole blob
        ax[1].add_patch(d)
    # ax[0].set_axis_off()
    plt.tight_layout()
    if fname:
        plt.savefig(fname)
    else:
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
    print("Filtered blobs:", len(flitered_blobs))
    return flitered_blobs

def flat_label_filter(flats, label_filter = 1):
    if (label_filter != 'na'):
        filtered_idx = flats[:, 3] == label_filter
        flats = flats[filtered_idx, :]
    return (flats)


def crop_blobs(blobs, block_img, area=-1, place_holder=-1, crop_width=100,
               blob_extention_ratio=1.4, blob_extention_radius=2):
    '''
    input1: blobs, blob info [n, 0:3] [y, x, r]
    input2: block_img, corresponding block_img
    plt: cropped images
    return: cropped padded images in a flattened 2d-array, with meta data in the first 6 numbers
            - rows: blobs
            - columns: [y, x, r_, label, area, place_holder, flattened cropped_blob_img (crop_width^2)]
                - y, x: corrd of centroid on original block_img
                - r_: extended radius
                - label: -1 = unlabeled; 1 = yes, 0 = no
                - area: in pixels, -1 as na
                - place_holder: for future use

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
                              [y, x, r_, -1, area, place_holder])  # -1 unlabeled
        flat_crop = np.array([flat_crop])
        flat_crops = np.append(flat_crops, flat_crop, axis=0)
    return flat_crops


def mask_image(image, r = 10, blob_extention_ratio=1, blob_extention_radius=0):
    '''
    input: one image [100, 100], and radius of the blob
    return: hard-masked image
    '''
    r_ = r * blob_extention_ratio + blob_extention_radius
    w = int(image.shape[0]/2)

    # hard mask creating training data
    mask = np.zeros((2 * w, 2 * w))  # zeros are masked to be black
    rr, cc = circle(w - 1, w - 1, min(r_, w - 1))
    mask[rr, cc] = 1  # 1 is white
    hard_masked = (1 - (1 - image) * mask)

    return hard_masked


def reshape_img_from_flat(flat_crop):
    flat = flat_crop[6:]
    w = int(sqrt(len(flat)) / 2)
    image = np.reshape(flat, (w + w, w + w))
    return image


def plot_flat_crop(flat_crop, blob_extention_ratio=1, blob_extention_radius=0, fname=None, plot_area=True):
    '''
    input: one padded crop of a blob
    plt: yellow circle and hard-masked side-by-side
    columns: [y, x, r_, area, place_holder, flattened cropped_blob_img (crop_width^2)]
    return: cropped image and hard-masked image
    '''
    # reshape
    [y, x, r, label, area, place_holder] = flat_crop[0:6]
    r_ = r * blob_extention_ratio + blob_extention_radius

    flat = flat_crop[6:]
    w = int(sqrt(len(flat)) / 2)
    image = np.reshape(flat, (w + w, w + w))
    #print("max_pixel value:", round(np.max(image), 3))

    # Equalized
    equalized = equalize(image)

    # hard mask creating training data
    # hard_masked = mask_image(equalized, r=r_)

    area_plot = area_calculation(image, r, plotting=True)

    fig, axes = plt.subplots(1, 4, figsize=(8, 32), sharex=False, sharey=False)
    ax = axes.ravel()
    ## Auto Contrast For labeler
    ax[0].set_title('For Labeling\ncurrent label:{}'.format(int(label)))
    ax[0].imshow(image, 'gray')
    c = plt.Circle((w - 1, w - 1), r_, color=(0.9, 0.9, 0, 0.5), linewidth=1, fill=False)
    ax[0].add_patch(c)

    ## Original for QC
    ax[1].set_title('Native Contrast\nblob_detection radius:{}'.format(r))
    ax[1].imshow(image, 'gray', clim=(0.0, 1.0))
    c = plt.Circle((w - 1, w - 1), r_, color=(0.9, 0.9, 0, 0.5), linewidth=1, fill=False)
    ax[1].add_patch(c)

    ## Equalized for QC
    ax[2].set_title('Equalized\narea (pixels):{}'.format(int(area)))
    ax[2].imshow(equalized, 'gray', clim=(0.0, 1.0))
    c = plt.Circle((w - 1, w - 1), r_, color=(0.9, 0.9, 0, 0.5), linewidth=1, fill=False)
    ax[2].add_patch(c)

    ## area calculation
    ax[3].set_title('Area Calculation\narea (pixels):{}'.format(int(area)))
    ax[3].imshow(area_plot, 'gray', clim=(0.0, 1.0))
    c = plt.Circle((w - 1, w - 1), r_, color=(0.9, 0.9, 0, 0.5), linewidth=1, fill=False)
    ax[3].add_patch(c)

    plt.tight_layout()
    if fname:
        plt.savefig(fname+".png")
    else:
        plt.show()
    fig.canvas.draw()

    return True

def plot_flat_crops(flat_crops, blob_extention_ratio=1, blob_extention_radius=0, fname=None, plot_area=False):
    '''
    input: flat_crops
    task: plot padded crop and hard-masked crop side-by-side
    '''
    for i, flat_crop in enumerate(flat_crops):
        if fname:
            plot_flat_crop(flat_crop, blob_extention_ratio=blob_extention_ratio, blob_extention_radius=blob_extention_radius, 
                fname=fname+'.rnd'+str(i), plot_area=plot_area)
        else:
            plot_flat_crop(flat_crop, blob_extention_ratio=blob_extention_ratio, blob_extention_radius=blob_extention_radius,
                plot_area=plot_area)




def area_calculation(img, r, plotting=False, fname=None):
    #todo: increase speed
    from skimage import io, filters
    from scipy import ndimage
    import matplotlib.pyplot as plt
    
    # automatic thresholding method such as Otsu's (avaible in scikit-image)
    img = equalize(img)  # no use
    img = normalize_img(img)  # bad
    # val = filters.threshold_otsu(img)
    try:
        val = filters.threshold_yen(img)
    except ValueError: 
        #print("Ops, got blank blob crop")
        return (0)

    # val = filters.threshold_li(img)

    drops = ndimage.binary_fill_holes(img < val)  # cells as 1 (white), bg as 0
    
    # create mask 
    w = int(img.shape[0]/2)
    mask = np.zeros((2 * w, 2 * w))  # zeros are masked to be black
    rr, cc = circle(w - 1, w - 1, min(r, w - 1))
    mask[rr, cc] = 1  # 1 is white
    
    # apply mask on binary image
    drops = abs(drops * mask)
    
    if (plotting):
        plt.subplot(1, 2, 1)
        plt.imshow(img, 'gray', clim=(0, 1))
        plt.subplot(1, 2, 2)
        plt.imshow(drops, cmap='gray')
        if fname:
            plt.savefig(fname+'.png')
        else:
            plt.show()
    #         plt.hist(drops.flatten())
    #         plt.show()
        #print('intensity cut-off is', round(val, 3), '; pixcel count is %d' %(int(drops.sum())))
        return drops
    else:
        return int(drops.sum())


def show_rand_crops(crops, label_filter="na", num_shown=5, 
    blob_extention_ratio=1, blob_extention_radius=0, 
    plot_area=False, seed = None, fname=None):
    '''
    blobs: the blobs crops
    label_filter: 0, 1, -1; "na" means no filter
    fname: None, plot.show(); if fname provided, saved to png
    '''
    if (label_filter != 'na'):
        filtered_idx = [str(int(x)) == str(label_filter) for x in crops[:, 3]]
        #print('labels:',[str(int(x)) for x in crops[:, 3]])
        #print('filter:', filtered_idx)
        crops = crops[filtered_idx, :]


    if (len(crops) >= num_shown):
        print(num_shown, "blobs will be plotted")
        if seed:
            np.random.seed(seed)
        randidx = np.random.choice(range(len(crops)), num_shown, replace=False)
        np.random.seed()

        if (plot_area):
            Images, Labels, Rs = parse_blobs(crops[randidx, :])
            [area_calculation(image, r=Rs[ind], plotting=True) for ind, image in enumerate(Images)]

        plot_flat_crops(crops[randidx, :], 
            blob_extention_ratio=blob_extention_ratio, blob_extention_radius=blob_extention_radius, fname=fname)
    elif (len(crops) > 0):
        print("Only", len(crops), 'blobs exist, and will be plotted')
        plot_flat_crops(crops,
            blob_extention_ratio=blob_extention_ratio, blob_extention_radius=blob_extention_radius, fname=fname)

        if (plot_area):
            Images, Labels, Rs = parse_blobs(crops)
            [area_calculation(image, r=Rs[ind], plotting=True) for ind, image in enumerate(Images)]
    else:
        print('num_blobs after filtering is 0')
        
    return (True)


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
            if bottom - top < 10 or right - left < 10:
                image_equ[top:bottom, left:right] = image[top:bottom, left:right] # skip equalization
            else:
                image_equ[top:bottom, left:right] = equalize(image[top:bottom, left:right]) # for each block
            c += 1
        
        # For right most columns
        left = c * block_width
        right = image.shape[1]
        if bottom - top < 10 or right - left < 10:
            image_equ[top:bottom, left:right] = image[top:bottom, left:right] # skip equalization
        else:
            image_equ[top:bottom, left:right] = equalize(image[top:bottom, left:right]) # for each block
        
        r += 1
    
    # For bottom row
    top = r * block_height
    bottom = image.shape[0]
    c = 0
    while (c + 1) * block_width <= image.shape[1]:
        # get each block
        left = c * block_width
        right = (c + 1) * block_width
        if bottom - top < 10 or right - left < 10:
            image_equ[top:bottom, left:right] = image[top:bottom, left:right] # skip equalization
        else:
            image_equ[top:bottom, left:right] = equalize(image[top:bottom, left:right]) # for each block
        c += 1

    left = c * block_width
    right = image.shape[1]
    if bottom - top < 10 or right - left < 10:
        image_equ[top:bottom, left:right] = image[top:bottom, left:right] # skip equalization
    else:
        image_equ[top:bottom, left:right] = equalize(image[top:bottom, left:right]) # for each block

    return image_equ


def find_blobs_and_crop(image, image_equ, 
    crop_width=100,
    scaling_factor=2,
    max_sigma=40, min_sigma=11, num_sigma=5, threshold=0.1, overlap=.0,
    blob_extention_ratio=1.4, blob_extention_radius=2,
    fname=None
    ):
    '''
    detect with image_equ
    return blobs with image
    if fname assigned xx.jpg, will save to jpg rather than plot inline
    '''
    print("scaling factor for findinb_blobs is ", scaling_factor)
    
    image_flat_crops = np.empty((0, int(6 + 2 * crop_width * 2 * crop_width)))

    print("Finding blobs")
    blobs = find_blob(
        (1 - image_equ), scaling_factor=scaling_factor,
        max_sigma=max_sigma, min_sigma=min_sigma, 
        num_sigma=num_sigma, threshold=threshold, overlap=overlap
    )  # reverse color, get bright blobs
    
    print(blobs.shape, "detected")

#     print("Visualizing blobs")
#     vis_blob_on_block(blobs, image_equ, image, 
#         blob_extention_ratio=blob_extention_ratio, 
#         blob_extention_radius=blob_extention_radius, 
#                      fname=fname)

    # create crop from blob
    print("Creating crops from blobs")
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
yes=1, no=0, undistinguishable=3, skip=s, go-back=b, excape(pause)=e: '''.format(i + 1, num_unlabeled))

        if label == '1':
            flat_crops[idx[i], 3] = 1  # yes
        elif label == '0':
            flat_crops[idx[i], 3] = 0  # no
        elif label == '3':
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


def sub_sample(A, n, seed=1):
    if n < A.shape[0]:
        np.random.seed(seed=seed)
        A = A[np.random.choice(A.shape[0], n, replace=False), :]
        np.random.seed(seed=None)
    else:
        pass
    return (A)
    

def load_blobs_db(in_db_name, n_subsample=False, seed=1):
    '''
    use parameters: db_name
    input: db fname from user (xxx.npy)
    output: array (crops format)
    '''
    import gzip
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
        blobs_stat(image_flat_crops)
        
        if n_subsample:
            print("subsampling to", n_subsample, "blobs")
            image_flat_crops = sub_sample(image_flat_crops, n_subsample, seed=seed)  
    else:
        print("{} file not found".format(in_db_name))

    return image_flat_crops

def save_blobs_db(crops, fname):
    import subprocess
    fname = fname.replace(".npy.gz", ".npy")
    np.save(fname, crops)
    subprocess.run("gzip -f " + fname, shell=True, check=True)


def remove_edge_crops(flat_blobs):
    """
    some crops of blobs contain edges, because they are from the edge of scanned areas or on the edge of the well
    use this function to remove blobs with obvious long straight black/white lines
    """
    import cv2
    good_flats = []
    for i in range(0, flat_blobs.shape[0]):
        flat = flat_blobs[i,]
        crop = reshape_img_from_flat(flat)
        crop = crop * 255
        crop = crop.astype(np.uint8)
    
        crop = cv2.blur(crop,(4,4))
    
        edges = cv2.Canny(crop,50,150,apertureSize = 3)

        minLineLength = 40
        maxLineGap = 10
        lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength,maxLineGap)
    
        if lines is not None: # has lines
            pass
#             print(lines.shape)
#             for i in range(0, lines.shape[0]):
#                 for x1,y1,x2,y2 in lines[i]:
#                     cv2.line(edges,(x1,y1),(x2,y2),(255,255,0, 0.8),6)
#             plt.title("Bad")
#             plt.imshow(crop)
#             plt.show()
        else: # no lines
            good_flats.append(flat)
#             plt.title(str(i))
#             plt.imshow(crop, 'gray')
#             plt.show()
    #         plt.imshow(edges, "gray")
    #         plt.title(str(i))
    #         plt.show()
    #         print("Good")
    
    good_flats = np.stack(good_flats)
    return (good_flats)




def sample_crops(crops, proportion, seed):
    np.random.seed(seed)
    crops = np.random.permutation(crops)
    sample = crops[range(int(len(crops)*proportion)), :]
    np.random.seed(seed=None)
    print(len(sample), "samples taken")
    return sample
    

# TEST SCALING and Equalization
# i = 0; j = 0; l = 2048
# block = image[2048*i : 2048*(i+1), 
#               2048*j : 2048*(j+1)]
# block_equ = equalize(block)
# block_equ_small = down_scale(block_equ, 2)

# print('block image: ', block.shape)
# print ("resized image: ", block_equ_small.shape)

# fig, axes = plt.subplots(1, 3, figsize=(20, 60), sharex=False, sharey=False)
# ax = axes.ravel()show_rand_crops
# ax[0].set_title('input block')
# ax[0].imshow(block, 'gray')
# ax[1].set_title('equalized block')
# ax[1].imshow(block_equ, 'gray')
# ax[2].set_title('scaled block')
# ax[2].imshow(block_equ_small, 'gray')

def split_train_valid(array, training_ratio):
    """
    Split into train and valid
    :param array: 2D array, each row is a sample
    :param ratio: ratio of train in all, e.g. 0.7
    :return: two arrays
    """
    N = array.shape[0]
    N1 = int(N * training_ratio)
    np.random.seed(3)
    np.random.shuffle(array)
    np.random.seed()
    train = array[0:N1]
    valid = array[N1:]
    return train, valid

def normalize_img(image):
    '''
    Normalize images into [0,1]
    :param image:
    :return:
    '''
    image = image - np.min(image)
    image = image / np.max(image)
    return image


def balancing_by_removing_no(blobs):
    '''
    balance yes/no ratio to 1, by removing excess NO samples
    :return: balanced blobs (with less samples)
    '''
    print('Before balancing:')
    blobs_stat(blobs)

    idx_yes = np.arange(0, blobs.shape[0])[blobs[:, 3] == 1]
    idx_no = np.arange(0, blobs.shape[0])[blobs[:, 3] == 0]
    N_Yes = len(idx_yes)
    N_No = len(idx_no)

    if N_No > N_Yes:
        print('number of No matched to yes by sub-sampling')
        idx_no = np.random.choice(idx_no, N_Yes, replace=False)
        idx_choice = np.concatenate([idx_yes, idx_no])
        np.random.seed(2)
        np.random.shuffle(idx_choice)
        np.random.seed()
        blobs = blobs[idx_choice,]

    print("After balancing by removing neg samples")
    blobs_stat(blobs)

    return blobs


def balancing_by_duplicating(blobs):
    '''
    balance yes/no ratio to 1, by duplicating blobs in the under-represented group
    only yes/no considered
    undistinguishable not altered
    result randomized to avoid problems in training
    :return: balanced blobs (with less samples)
    '''
    print('Before balancing:')
    blobs_stat(blobs)

    idx_yes = np.arange(0, blobs.shape[0])[blobs[:, 3] == 1]
    idx_no = np.arange(0, blobs.shape[0])[blobs[:, 3] == 0]
    idx_unsure = np.arange(0, blobs.shape[0])[blobs[:, 3] == -2]
    N_Yes = len(idx_yes)
    N_No = len(idx_no)
    N_unsure = len(idx_unsure)

    # todo: include unsure
    if N_No > N_Yes:
        print('number of No matched to Yes by re-sampling')
        idx_yes = np.random.choice(idx_yes, N_No, replace=True)  # todo: some yes data lost when N_No small
    elif N_Yes > N_No:
        print('number of Yes matched to No by re-sampling')
        idx_no = np.random.choice(idx_no, N_Yes, replace=True)
    idx_choice = np.concatenate([idx_yes, idx_no, idx_unsure])  # 3 classes
    np.random.shuffle(idx_choice)
    blobs = blobs[idx_choice, ]

    print("After balancing by adding positive samples")
    blobs_stat(blobs)

    return blobs


def parse_blobs(blobs):
    '''
    parse blobs into Images, Labels, Rs
    :param blobs:
    :return:  Images, Labels, Rs
    '''
    Flats = blobs[:, 6:]
    w = int(sqrt(blobs.shape[1] - 6) / 2)  # width of img
    Images = Flats.reshape(len(Flats), 2*w, 2*w)
    Labels = blobs[:, 3]
    Rs = blobs[:, 2]

    return Images, Labels, Rs


def augment_images(Images, aug_sample_size):
    '''
    Input images (n_samples, 2*w, 2*w)
    Process: Augmentation; Normalization back to [0, 1]
    Output augmented images of the same shape
    :param Images:
    :return: augImages
    '''
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    import imgaug as ia
    from imgaug import augmenters as iaa


    sometimes = lambda aug: iaa.Sometimes(0.9, aug)

    w2 = Images.shape[1]
    Images = Images.reshape(len(Images), w2, w2, 1) # formatting

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.5), # vertically flip 20% of all images
            sometimes(iaa.Affine(
                # todo: more strict; no scaling down
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                rotate=(-90, 90), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 1), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
        ],
        random_order=True
    )

    Images_ = seq.augment_images(Images)
    while Images_.shape[0] < aug_sample_size:
        _ = seq.augment_images(Images)
        Images_ = np.vstack((Images_, _))
    Images = Images_[0:aug_sample_size, :]
    print('shape:', Images.shape, 'after augment_images')

    Images = Images.reshape(len(Images), w2, w2)  # formatting back
    Images = np.array([normalize_img(image) for image in Images])
    return Images


def F1(y_pred, y_true):
    from keras import backend as K
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    f1 = 2*((precision*recall)/(precision+recall+K.epsilon()))
    return f1

def f1_score(precision, recall):
    return 2*(precision*recall/(precision+recall+1e-07))

def F1_calculation(predictions, labels):
    print("F1_calculation for sure labels only")
    idx = (labels == 1) | (labels == 0)  # sure only
    labels = labels[idx, ]
    predictions = predictions[idx, ]


    TP = np.sum(np.round(predictions * labels))
    PP = np.sum(np.round(labels))
    recall = TP / (PP + 1e-7)

    PP2 = np.sum(np.round(predictions))
    precision = TP/(PP2 + 1e-7)

    F1 = 2*((precision*recall)/(precision+recall+1e-7))

    print('Precition: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(precision*100, recall*100, F1*100))

    return F1


def preprocessing_imgs(Images, Rs, Labels, scaling_factor):
    # Downscale images (todo: downscale as the first step)
    print("Downscaling images by ", scaling_factor)
    Images = np.array([down_scale(image, scaling_factor=scaling_factor) for image in Images])
    ## Downscale w and R
    print('w after scaling:', w)
    Rs = Rs/scaling_factor

    # Equalize images (todo: test equalization -> scaling)
    # todo: more channels (scaled + equalized + original)
    print("Equalizing images...")
    # todo:  Possible precision loss when converting from float64 to uint16
    Images = np.array([equalize(image) for image in Images])

    # Mask images
    print("Masking images...")
    Images = np.array([mask_image(image, r=Rs[ind]) for ind, image in enumerate(Images)])

    # Normalizing images
    print("Normalizing images...")
    # todo:  Possible precision loss when converting from float64 to uint16
    Images = np.array([normalize_img(image) for image in Images])

    return Images, Rs, Labels, w


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE  # single core
#from MulticoreTSNE import MulticoreTSNE as TSNE  # MCORE

def cluster_scatterplot(df2d, labels, title):
    '''
    PCA or t-SNE 2D visualization

    `cluster_scatterplot(tsne_projection, cluster_info.Cluster.values.astype(int),
                    title='projection.csv t-SNE')`

    :param df2d: PCA or t-SNE projection df, cell as row, feature as columns
    :param labels:
    :param title:
    :return:
    '''
    legends = np.unique(labels)
    print('all labels:', legends)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)

    for i in legends:
        _ = df2d.iloc[labels == i]
        num_blobs = str(len(_))
        percent_cells = str(round(int(num_blobs) / len(df2d) * 100, 1)) + '%'
        ax.scatter(_.iloc[:, 0], _.iloc[:, 1],
                   alpha=0.5, marker='.',
                   label='c' + str(i) + ':' + num_blobs + ', ' + percent_cells
                   )

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.xlabel('legend format:  cluster_id:num-cells')

    #plt.savefig(title + '.png', bbox_inches='tight')
    plt.show()
    plt.close('all')


def pca_tsne(df_gene_col, cluster_info=None, title='data', 
             #dir='plots',
             num_pc=50, num_tsne=2, ncores=8):
    '''
    PCA and tSNE plots for DF_cell_row, save projections.csv
    :param df_cell_row: data matrix, features as columns, e.g. [cell, gene]
    :param cluster_info: cluster_id for each cell_id
    :param title: figure title, e.g. Late
    :param num_pc: 50
    :param num_tsne: 2
    :return: tsne_df, plots saved, pc_projection.csv, tsne_projection.csv saved
    '''

#     if not os.path.exists(dir):
#         os.makedirs(dir)

#     title = './' + dir + '/' + title

    df = df_gene_col
    if cluster_info is None:
        cluster_info = pd.DataFrame(0, index=df.index, columns=['cluster_id'])

    tic = time.time()
    # PCA
    pca = PCA(n_components=num_pc)
    pc_x = pca.fit_transform(df)
    df_pc_df = pd.DataFrame(data=pc_x, index=df.index, columns=range(num_pc))
    df_pc_df.index.name = 'cell_id'
    df_pc_df.columns.name = 'PC'
    #df_pc_df.to_csv(title + '.pca.csv')
    print('dim before PCA', df.shape)
    print('dim after PCA', df_pc_df.shape)
    print('explained variance ratio: {}'.format(
        sum(pca.explained_variance_ratio_)))

    colors = cluster_info.reindex(df_pc_df.index)
    colors = colors.dropna().iloc[:, 0]
    print('matched cluster_info:', colors.shape)
    print('unmatched data will be excluded from the plot')  # todo: include unmatched

    df_pc_ = df_pc_df.reindex(colors.index)  # only plot labeled data?
    cluster_scatterplot(df_pc_, colors.values.astype(str), title=title + ' (PCA)')

#     # tSNE
#     print('MCORE-TSNE, with ', ncores, ' cores')
#     df_tsne = TSNE(n_components=num_tsne, n_jobs=ncores).fit_transform(df_pc_)
#     print('tsne done')
#     df_tsne_df = pd.DataFrame(data=df_tsne, index=df_pc_.index)
#     print('wait to output tsne')
#     df_tsne_df.to_csv(title + '.tsne.csv')
#     print('wrote tsne to output')
#     cluster_scatterplot(df_tsne_df, colors.values.astype(str), title=title + ' ('
#                                                                              't-SNE)')
    toc = time.time()
    print('took {:.1f} seconds\n'.format(toc - tic))

    return df_pc_df

def input_names(SAMPLES, words = ["Top", "Left", "Right", "Bottom"], 
    prefix = "res/blobs/view/", suffix = '.html', NUMS=[0,1,2,3]):
    """
    If "Top", only one output
    else four output

    return ['res/blobs/view/s1.0.html', 'res/blobs/view/s1.1.html', 2, 3,
    'res/blobs/view/s2.0.html', ..]

    ['res/blobs/view/E2f4_CFUe_WT3_3-Stitching-20.0.html', 'res/blobs/view/E2f4_CFUe_WT3_3-Stitching-20.1.html', 'res/blobs/view/E2f4_CFUe_WT3_3-Stitching-20.2.html', 'res/blobs/view/E2f4_CFUe_WT3_3-Stitching-20.3.html', 'res/blobs/view/E2f4_CFUe_WT3_3_Top-Stitching-21.0.html']
    """
    # expand("res/blobs/view/{s}.{i}.html", s=SAMPLES, i=NUMS),  # rand samples of detected blobs
    lst = []
    for s in SAMPLES:
        if any([w in s for w in words]):
            res = prefix + s + '.0' + suffix
            lst.append(res)
        else:
            res = map(lambda i: prefix + s + '.' + str(i) + suffix, NUMS)
            lst=lst+list(res)
    return lst

def get_samples(DATA_DIR):
    """
    input: 'data/' or 'data', the path of czi files
    e.g. 
    E2f4_CFUe_KO_1-Stitching-01.czi       E2f4_CFUe_WT2_1-Stitching-12.czi
    E2f4_CFUe_KO_2-Stitching-02.czi       E2f4_CFUe_WT2_1_Top-Stitching-13.czi
    E2f4_CFUe_KO_3-Stitching-03.czi       E2f4_CFUe_WT2_2-Stitching-14.czi
    E2f4_CFUe_NoEpo_1-Stitching-04.czi    E2f4_CFUe_WT2_3-Stitching-15.czi
    E2f4_CFUe_NoEpo_2-Stitching-05.czi    E2f4_CFUe_WT3_1-Stitching-16.czi
    E2f4_CFUe_NoEpo_3-Stitching-06.czi    E2f4_CFUe_WT3_1_Top-Stitching-17.czi
    E2f4_CFUe_WT1_1-Stitching-07.czi      E2f4_CFUe_WT3_2-Stitching-18.czi
    E2f4_CFUe_WT1_2-Stitching-08.czi      E2f4_CFUe_WT3_2_Top-Stitching-19.czi
    E2f4_CFUe_WT1_2_Top-Stitching-09.czi  E2f4_CFUe_WT3_3-Stitching-20.czi
    E2f4_CFUe_WT1_3-Stitching-10.czi      E2f4_CFUe_WT3_3_Top-Stitching-21.czi
    E2f4_CFUe_WT1_3_Top-Stitching-11.czi  yung.txt

    output: 
    ['E2f4_CFUe_KO_1-Stitching-01',
     'E2f4_CFUe_KO_2-Stitching-02',
     'E2f4_CFUe_KO_3-Stitching-03',
     'E2f4_CFUe_NoEpo_1-Stitching-04',
     'E2f4_CFUe_NoEpo_2-Stitching-05',
     'E2f4_CFUe_NoEpo_3-Stitching-06',
     'E2f4_CFUe_WT1_1-Stitching-07',
     'E2f4_CFUe_WT1_2-Stitching-08',
     'E2f4_CFUe_WT1_2_Top-Stitching-09',
     'E2f4_CFUe_WT1_3-Stitching-10',
     'E2f4_CFUe_WT1_3_Top-Stitching-11',
     'E2f4_CFUe_WT2_1-Stitching-12',
     'E2f4_CFUe_WT2_1_Top-Stitching-13',
     'E2f4_CFUe_WT2_2-Stitching-14',
     'E2f4_CFUe_WT2_3-Stitching-15',
     'E2f4_CFUe_WT3_1-Stitching-16',
     'E2f4_CFUe_WT3_1_Top-Stitching-17',
     'E2f4_CFUe_WT3_2-Stitching-18',
     'E2f4_CFUe_WT3_2_Top-Stitching-19',
     'E2f4_CFUe_WT3_3-Stitching-20',
     'E2f4_CFUe_WT3_3_Top-Stitching-21']
    """
    import os
    import re
    SAMPLES=os.listdir(DATA_DIR)
    SAMPLES=list(filter(lambda x: x.endswith("czi"), SAMPLES))
    SAMPLES=[re.sub(".czi", "", x) for x in SAMPLES]
    return SAMPLES
