def mask_image(image, r = 10, blob_extention_ratio=1, blob_extention_radius=0):
    '''
    input: one image [100, 100], and radius of the blob
    return: hard-masked image of [0,1] scale
    '''
    import numpy as np
    from skimage.draw import disk
    from ..img.auto_contrast import float_image_auto_contrast
    
    image = float_image_auto_contrast(image)

    r_ = r * blob_extention_ratio + blob_extention_radius
    w = int(image.shape[0]/2)

    # hard mask creating training data
    mask = np.zeros((2 * w, 2 * w))  # zeros are masked to be black
    rr, cc = disk((w - 1, w - 1), min(r_, w - 1))
    mask[rr, cc] = 1  # 1 is white
    hard_masked = (1 - (1 - image) * mask)

    return hard_masked
