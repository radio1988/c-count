import warnings
from skimage import exposure
import numpy as np

def equalize(image):
    '''
    input: image: 2d-array
    output: image: 2d-array, 0 black, 1 white
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    An algorithm for local contrast enhancement, that uses histograms computed over different tile regions of the image.
    Local details can therefore be enhanced even in regions that are darker or lighter than most of the image.
    '''
    warnings.filterwarnings("ignore")
    return exposure.equalize_adapthist(image, clip_limit=0.01)  # Aug, 2019, cleaner image than 0.03


def block_equalize(image, block_height=2048, block_width=2048):
    '''
    split
    equalization
    stitch and return
    '''
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
