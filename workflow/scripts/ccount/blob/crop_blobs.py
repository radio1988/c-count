def crop_blobs(blobs, block_img, area=-1, place_holder=-1, crop_width=100,
               blob_extention_ratio=1.4, blob_extention_radius=2):
    '''
    input1: blobs, blob info [n, 0:3] [y, x, r]
    input2: block_img, corresponding block_img
    plt: cropped images
    return: cropped padded images in a flattened 2d-array, with meta data in the first 6 numbers
            - rows: blobs
            - columns: [y, x, r_, label, area, place_holder, flattened cropped_blob_img (crop_width^2)]
                - y, x: corrd of centroid on original block_img
                - r_: extended radius
                - label: -1 = unlabeled; 1 = yes, 0 = no
                - area: in pixels, -1 as na
                - place_holder: for future use

    Algorithm:
    1. White padding
    2. Crop for each blob
    '''
    import numpy as np

    # White padding so that blobs on the edge can get cropped image
    padder = max(np.max(block_img), 1)
    padded = np.pad(block_img, crop_width, pad_with, padder=padder)  # 1 = white padding, 0 = black padding

    # crop for each blob
    flat_crops = np.empty((0, int(6 + 2 * crop_width * 2 * crop_width)))
    for blob in blobs:
        y, x, r = blob  # conter-intuitive order
        # print("the {}th blob  x:{} y:{} r:{}\n".format(i, x, y, r))
        y_ = int(y + crop_width)
        x_ = int(x + crop_width)  # adj for padding
        r_ = int(
            blob_extention_ratio * r + blob_extention_radius)  # extend circles, need to keep parameters sync between functions

        cropped_img = padded[y_ - crop_width: y_ + crop_width,
                      x_ - crop_width: x_ + crop_width]  # x coordinates use columns to locate, vise versa

        flat_crop = np.insert(cropped_img.flatten(), [0, 0, 0, 0, 0, 0],
                              [y, x, r_, -1, area, place_holder])  # -1 unlabeled
        flat_crop = np.array([flat_crop])
        flat_crops = np.append(flat_crops, flat_crop, axis=0)
    return flat_crops