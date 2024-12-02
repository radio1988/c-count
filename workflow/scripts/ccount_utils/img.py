import warnings
import numpy as np
from skimage import exposure
from aicsimageio import AICSImage
from skimage.transform import rescale, resize, downscale_local_mean


def equalize(image):
    """
    input: image: 2d-array
    output: image: 2d-array, 0 black, 1 white
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    An algorithm for local contrast enhancement, that uses histograms computed over different tile regions of the image.
    Local details can therefore be enhanced even in regions that are darker or lighter than most of the image.
    """
    warnings.filterwarnings("ignore")
    return exposure.equalize_adapthist(image, clip_limit=0.01)  # Aug, 2019, cleaner image than 0.03


def block_equalize(image, block_height=2048, block_width=2048):
    """
    split
    equalization
    stitch and return
    """
    image_equ = np.empty(image.shape)
    if block_width <= 0:
        return equalize(image)

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
                image_equ[top:bottom, left:right] = image[top:bottom, left:right]  # skip equalization
            else:
                image_equ[top:bottom, left:right] = equalize(image[top:bottom, left:right])  # for each block
            c += 1

        # For right most columns
        left = c * block_width
        right = image.shape[1]
        if bottom - top < 10 or right - left < 10:
            image_equ[top:bottom, left:right] = image[top:bottom, left:right]  # skip equalization
        else:
            image_equ[top:bottom, left:right] = equalize(image[top:bottom, left:right])  # for each block

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
            image_equ[top:bottom, left:right] = image[top:bottom, left:right]  # skip equalization
        else:
            image_equ[top:bottom, left:right] = equalize(image[top:bottom, left:right])  # for each block
        c += 1

    left = c * block_width
    right = image.shape[1]
    if bottom - top < 10 or right - left < 10:
        image_equ[top:bottom, left:right] = image[top:bottom, left:right]  # skip equalization
    else:
        image_equ[top:bottom, left:right] = equalize(image[top:bottom, left:right])  # for each block

    return image_equ


def uint16_image_auto_contrast(image):
    """
    pos image
    output forced into uint16
    max_contrast_achieved
    for 2019 format, input is also uint16
    """
    image = image - np.min(image)  # pos image
    image = image / np.max(image) * (2 ** 16 - 1)
    image = image.astype(np.uint16)
    return image


def float_image_auto_contrast(image):
    """
    Normalize images into [0,1]
    :param image:
    :return:
    """
    image = image - np.min(image)
    image = image / np.max(image)
    image = image.astype(np.float16)
    return image


def read_czi(fname, Format="2019"):
    """
    input: fname of czi file
    output: image_obj
    """
    fname = str(fname)
    Format = str(Format)
    print('read_czi:', fname)
    print('Format', Format)
    if fname.endswith('czi'):
        if Format == '2019':
            image_obj = AICSImage(fname)
        else:
            raise Exception("Format not accepted")
    elif fname.endswith('czi.gz'):
        raise Exception("todo")
    else:
        raise Exception("input czi/czi.gz file type error\n")

    return image_obj


def parse_image_obj(image_obj, i=0, Format='2019'):
    """
    input: image_obj
    output: image 2d np.array
    """
    Format = str(Format)
    i = int(i)
    if Format == '2019':
        image_obj.set_scene(i)
        image_array = image_obj.get_image_data()
        image = image_array[0, 0, 0, :, :]
    else:
        raise Exception("Format not accepted")
    return image


def down_scale(img, scaling_factor=2):
    """
    input1: image
    input2: scaling_factor, # scale factor for each dim 1-> 1/1, 2 -> 1/2, 4 -> 1/4
    return: down scaled image array that is 1/scale-ratio in both x and y dimentions
    e.g.: block_small = down_scale(block, scaling_factor)
    """
    return resize(img, (img.shape[0] // scaling_factor, img.shape[1] // scaling_factor))
