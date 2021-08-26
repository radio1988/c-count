def uint16_image_auto_contrast(image):
    '''
    pos image
    output forced into uint16
    max_contrast_achieved
    for 2019 format, input is also uint16
    '''
    import numpy as np

    image = image - np.min(image)  # pos image
    image = image/np.max(image) * 2**16
    image = image.astype(np.uint16)
    return image