def augment_images(images, aug_sample_size):
    '''
    Input images (n_samples, 2*w, 2*w)
    Process: Augmentation; Normalization back to [0, 1]
    Output augmented images of the same shape
    :param images:
    :return: augimages
    '''
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    import imgaug as ia
    from imgaug import augmenters as iaa
    import numpy as np
    from ccount.blob.img.auto_contrast import float_image_auto_contrast

    sometimes = lambda aug: iaa.Sometimes(0.9, aug)

    w2 = images.shape[1]
    images = images.reshape(len(images), w2, w2, 1) # formatting

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.5), # vertically flip 20% of all images
            sometimes(iaa.Affine(
                # todo: more strict; no scaling down
                scale={(0.5, 2)}, # larger range hoping to eliminate size classifier
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                rotate=(-360, 360), 
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 1), # todo: [0, 255] for uint8 images
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
        ],
        random_order=True
    )

    #todo: sped up
    images_ = seq.augment_images(images)
    while images_.shape[0] < aug_sample_size:
        _ = seq.augment_images(images)
        images_ = np.vstack((images_, _))
    images = images_[0:aug_sample_size, :]
    print('shape:', images.shape, 'after augment_images')

    images = images.reshape(len(images), w2, w2)  # formatting back
    images = np.array([float_image_auto_contrast(image) for image in images])
    return images