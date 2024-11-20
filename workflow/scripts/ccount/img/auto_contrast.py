def uint16_image_auto_contrast(image):
    '''
    pos image
    output forced into uint16
    max_contrast_achieved
    for 2019 format, input is also uint16
    '''
    import numpy as np

    image = image - np.min(image)  # pos image
    image = image/np.max(image) * (2**16-1)
    image = image.astype(np.uint16)
    return image


def float_image_auto_contrast(image):
    '''
    Normalize images into [0,1]
    :param image:
    :return:
    '''
    import numpy as np
    
    image = image - np.min(image)
    image = image / np.max(image)
    image = image.astype(np.float16)
    return image