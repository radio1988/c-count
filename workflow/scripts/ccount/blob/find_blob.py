def find_blob(image_neg, scaling_factor = 2, 
    max_sigma=40, min_sigma=4, num_sigma=5, threshold=0.1, overlap=.0):
    '''
    input: gray scaled image with bright blob on dark background (image_neg)
    output: [n, 3] array of blob information, [y-locaiton, x-location, r-blob-radius] !!!
    https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html 
    https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log
    # larger num_sigma: more accurate boundry, slower, try 15
    # larger max_sigma: larger max blob size, slower
    # threshold: larger, less low contrast stuff
    '''
    from ..img.transform import down_scale
    from skimage.feature import  blob_log # blob_doh, blob_dog
    import time
    from math import sqrt

    print('image size', image_neg.shape)
    image_neg = down_scale(image_neg, scaling_factor)
    print('image-blob detection size', image_neg.shape)
    tic = time.time()
    blobs = blob_log(
        image_neg, 
        max_sigma=max_sigma, min_sigma=min_sigma, num_sigma=num_sigma, 
        threshold=threshold, overlap=overlap, exclude_border = False
        )
    blobs[:, 2] = blobs[:, 2] * sqrt(2)  # adjust r
    blobs = blobs * scaling_factor  # scale back coordinates
    toc = time.time()
    print("blob detection time: ", toc - tic)
    return blobs