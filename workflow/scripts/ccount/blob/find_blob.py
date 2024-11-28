def find_blob(image_neg, scaling_factor=4,
              max_sigma=12, min_sigma=3, num_sigma=20, threshold=0.1, overlap=.2):
    '''
    Input:
    gray scaled image with bright blob on dark background (image_neg)

    Output:
    [n, 3] array of blob information, [y, x, r]

    Method:
    https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html
    https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log

    Steps:
    1. scale down for faster processing
    2. blob detection
    3. scale back [y,x,r] and output

    Params:
    - max_sigma, min_sigma: the max/min size of blobs able to be detected
    - num_sigma: the accuracy, larger slower, more accurate
    - threshold: the min contrast for a blob
    - overlap: how much overlap of blobs allowed before merging

    # larger num_sigma: more accurate boundry, slower, try 15
    # larger max_sigma: larger max blob size, slower
    # threshold: larger, less low contrast stuff

    Default Params (Rui 2024)
    blob_detection_scaling_factor: 4  # 1, 2, 4 (08/23/2021 for blob_detection)
    max_sigma: 12 # 6 for 8 bit, larger for detecting larger blobs
    min_sigma: 3  # 2-8
    num_sigma: 20  # smaller->faster, less accurate, 5-20
    threshold: 0.1  # 0.02 too sensitive, 0.1 to ignore debris
    overlap: .2 # overlap larger than this, smaller blob gone, not sensitive
    blob_detection_visualization: True  # jpg file with circles around blobs
    blob_extention_ratio: 1.4 # for vis in jpg
    blob_extention_radius: 10 # for vis in jpg
    crop_width: 80  # padding width, which is cropped img width/2 (50), in blob_cropping.py
    '''
    from ..img.transform import down_scale
    from skimage.feature import blob_log  # blob_doh, blob_dog
    import time
    from math import sqrt

    print('image size:', image_neg.shape)

    image_neg = down_scale(image_neg, scaling_factor)
    print('scaled image size for blob detection:', image_neg.shape)

    tic = time.time()

    blobs = blob_log(
        image_neg,
        max_sigma=max_sigma, min_sigma=min_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap,
        exclude_border=False
    )

    blobs[:, 2] = blobs[:, 2] * sqrt(2)  # adjust r
    blobs = blobs * scaling_factor  # scale back coordinates

    toc = time.time()

    print("blob detection time: {}s".format(round(toc - tic), 2))
    print("{} blobs detected\n".format(blobs.shape[0]))
    return blobs
