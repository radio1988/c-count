def area_calculation(img, r, plotting=False, fname=None):
    #todo: increase speed
    from ..img.auto_contrast import float_image_auto_contrast
    from ..img.equalize import equalize
    from skimage import io, filters
    from skimage.draw import circle
    from scipy import ndimage
    import numpy as np
    import matplotlib.pyplot as plt

    
    # automatic thresholding method such as Otsu's (avaible in scikit-image)
    img = float_image_auto_contrast(img)  # bad
    img = equalize(img)  # no use

    try:
        val = filters.threshold_yen(img)
    except ValueError: 
        #print("Ops, got blank blob crop")
        return (0)

    drops = ndimage.binary_fill_holes(img < val)  # cells as 1 (white), bg as 0
    
    # create mask 
    w = int(img.shape[0]/2)
    mask = np.zeros((2 * w, 2 * w))  # zeros are masked to be black
    rr, cc = circle(w - 1, w - 1, min(r, w - 1))
    mask[rr, cc] = 1  # 1 is white
    
    # apply mask on binary image
    masked = abs(drops * mask)
    
    if (plotting):
        plt.subplot(1, 2, 1)
        plt.imshow(img, 'gray', clim=(0, 1))
        plt.subplot(1, 2, 2)
        plt.imshow(masked, cmap='gray')
        if fname:
            plt.savefig(fname+'.png')
        else:
            plt.show()

    return masked