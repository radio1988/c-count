def area_calculation(img, r, plotting=False, fname='blob_binary_image.png', 
                    blob_extention_ratio=1.4, blob_extention_radius=10):
    '''
    read one image
    output area-of-pixels as int
    '''
    #todo: increase speed
    from ..img.auto_contrast import float_image_auto_contrast
    from ..img.equalize import equalize
    from skimage import io, filters
    from skimage.draw import disk
    from scipy import ndimage
    import numpy as np
    import matplotlib.pyplot as plt

    
    # automatic thresholding method such as Otsu's (avaible in scikit-image)
    img = float_image_auto_contrast(img)  # bad
    # img = equalize(img)  # no use

    try:
        val = filters.threshold_yen(img)
    except ValueError: 
        #print("Ops, got blank blob crop")
        return (0)
    
    r = r * blob_extention_ratio + blob_extention_radius

    # cells as 1 (white), background as 0 (black)
    drops = ndimage.binary_fill_holes(img < val)  

    # mask out of the circle to be zero
    w = int(img.shape[0]/2)
    mask = np.zeros((2 * w, 2 * w)) 
    rr, cc = disk(w - 1, w - 1, min(r, w - 1))
    mask[rr, cc] = 1  # 1 is white
    
    # apply mask on binary image
    masked = abs(drops * mask)
    
    if (plotting):
        plt.subplot(1, 2, 1)
        plt.imshow(img, 'gray', clim=(0, 1))
        plt.subplot(1, 2, 2)
        plt.imshow(masked, cmap='gray')
        if fname:
            plt.savefig(fname)
        else:
            plt.show()

    return int(sum(sum(masked)))



def area_calculations(crops,
                              out_txt_name = "blobs.area.txt",
                              title="Blob Area In Pixcels",
                              plotting=True, txt_saving=True, crop_saving=True):

    '''only calculate for blobs matching the filter'''
    from ccount.blob.misc import parse_crops

    images, labels, rs = parse_crops(crops)
    areas = [area_calculation(image, r=rs[ind]) for ind, image in enumerate(images)]
    print("labels", [str(int(x)) for x in labels][0:min(5, len(labels))])
    print('areas:', areas[0:min(5, len(labels))])
    return (areas)




 
