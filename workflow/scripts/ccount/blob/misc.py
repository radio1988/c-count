## blob_locs and crops

def sub_sample(A, n, seed=1):
    if n <=0:
        raise Exception ('n must be float between 0-1 or int >=1')
    if n < 1:
        n = int(A.shape[0] * n)

    if n < A.shape[0]:
        np.random.seed(seed=seed)
        A = A[np.random.choice(A.shape[0], n, replace=False), :]
        np.random.seed(seed=None)
    else:
        pass
    return (A)


def blob_radius_histogram(blobs):
    '''
    show blob size distribution with histogram
    works on blob_locs or crops
    '''
    plt.title("Histogram of blob radius")
    plt.hist(crops[:, 2], 40)
    plt.show()


def blob_radius_filter(blobs, r_min, r_max):
    '''
    filter blobs based on size of r
    '''
    flitered_blobs = blobs[blobs[:, 2] >= r_min,]
    flitered_blobs = flitered_blobs[flitered_blobs[:, 2] < r_max,]
    print("Filtered blobs:", len(flitered_blobs))
    return flitered_blobs


## crops

def crops_stat(crops):
    '''
    print summary of labels in crops
    :param crops:
    :return:
    '''
    print("{} Yes, {} No, {} Uncertain, {} Unlabeled".format(
        sum(crops[:, 3] == 1),
        sum(crops[:, 3] == 0),
        sum(crops[:, 3] == -2),
        sum(crops[:, 3] == -1),))
    print("Total:", crops.shape[0])


def crop_width(image_flat_crops):
    from math import sqrt
    return  int(sqrt(crops.shape[1] - 6) / 2)


def parse_crops(crops):
    '''
    parse crops into images, labels, rs
    :param crops:
    :return:  images, labels, rs
    '''
    flats = crops[:, 6:]
    w = crop_width(crops)  # width of img
    images = flats.reshape(len(flats), 2*w, 2*w)
    labels = crops[:, 3]
    rs = crops[:, 2]

    return images, labels, rs


# def remove_edge_crops(flat_blobs):
#     """
#     some crops of blobs contain edges, because they are from the edge of scanned areas or on the edge of the well
#     use this function to remove blobs with obvious long straight black/white lines
#     """
#     import cv2
#     good_flats = []
#     for i in range(0, flat_blobs.shape[0]):
#         flat = flat_blobs[i,]
#         crop = flat2image(flat)
#         crop = crop * 255
#         crop = crop.astype(np.uint8)
    
#         crop = cv2.blur(crop,(4,4))
    
#         edges = cv2.Canny(crop,50,150,apertureSize = 3)

#         minLineLength = 40
#         maxLineGap = 10
#         lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength,maxLineGap)
    
#         if lines is not None: # has lines
#             pass
#         else: # no lines
#             good_flats.append(flat)
    
#     good_flats = np.stack(good_flats)
#     return (good_flats)


## flats    

def flat_label_filter(flats, label_filter = 1):
    if (label_filter != 'na'):
        filtered_idx = flats[:, 3] == label_filter
        flats = flats[filtered_idx, :]
    return flats


def flat2image(flat_crop):
    flat = flat_crop[6:]
    w = int(sqrt(len(flat)) / 2)
    image = np.reshape(flat, (w + w, w + w))
    return image


## images

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


def area_calculation(img, r, plotting=False, fname=None):
    #todo: increase speed
    from skimage import io, filters
    from scipy import ndimage
    import matplotlib.pyplot as plt
    
    # automatic thresholding method such as Otsu's (avaible in scikit-image)
    img = float_image_auto_contrast(img)  # bad
    img = equalize(img)  # no use

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
