from skimage.transform import rescale, resize, downscale_local_mean


def down_scale(img, scaling_factor=2):
    '''
    input1: image
    input2: scaling_factor, # scale factor for each dim 1-> 1/1, 2 -> 1/2, 4 -> 1/4
    return: down scaled image array that is 1/scale-ratio in both x and y dimentions
    e.g.: block_small = down_scale(block, scaling_factor)  
    '''
    return resize(img, (img.shape[0] // scaling_factor, img.shape[1] // scaling_factor))


def pad_with(vector, pad_width, iaxis, kwargs):
    '''
    to make np.pad in crop_blobs work
    '''
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector
