def vis_blob_detection_on_img(image, blob_locs, 
    blob_extention_ratio=1.0, blob_extention_radius=0, scaling = 2, fname=None):
    '''
    image: image where blobs were detected from
    blob_locs: blob info array n x 3 [x, y, r], crops also works, only first 3 columns used

    output: image with yellow circles around blobs
    '''
    from .transform import down_scale
    import matplotlib.pyplot as plt

    blob_locs = blob_locs[:, 0:3]
    blob_locs = blob_locs/scaling
    print('vis_blob_detection_on_img scaling:', scaling)

    image = down_scale(image, scaling)

    fig, axes = plt.subplots(1, 1, figsize=(40, 40), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].set_title('blob detection')
    ax[0].imshow(image, 'gray', interpolation='nearest')
    for loc in blob_locs:
        y, x, r = loc
        c = plt.Circle((x, y), 
                       r * blob_extention_ratio + blob_extention_radius, 
                       color=(0.9, 0.9, 0, 0.5), linewidth=1,
                       fill=False) 
        ax[0].add_patch(c)

    plt.tight_layout()
    if fname:
        plt.savefig(fname)
    else:
        plt.show()


def plot_flat_crop(flat_crop, blob_extention_ratio=1, blob_extention_radius=0, fname=None):
    '''
    input: one padded crop of a blob
    output: two plots
        - left: original image with yellow circle
        - right: binary for area calculation

    return: cropped image and hard-masked image
    '''
    import numpy as np
    from math import sqrt
    from .area_calculation import area_calculation
    from .misc import crop_width
    from ..img.transform import float_image_auto_contrast

    [y, x, r, label, area, place_holder] = flat_crop[0:6]
    r = r * blob_extention_ratio + blob_extention_radius
    image = flat2image(flat_crop)
    image = float_image_auto_contrast(image)
    w = crop_width(flat_crop)

    area_plot = area_calculation(image, r, plotting=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 16), sharex=False, sharey=False)
    ax = axes.ravel()
    ## Auto Contrast For labeler
    ax[0].set_title('Image for Labeling\ncurrent label:{}'.format(int(label)))
    ax[0].imshow(image, 'gray')
    c = plt.Circle((w - 1, w - 1), r, color=(0.9, 0.9, 0, 0.5), linewidth=1, fill=False)
    ax[0].add_patch(c)

    ## area calculation
    ax[1].set_title('Image for Area Calculation\narea (pixels):{}'.format(int(area)))
    ax[1].imshow(area_plot, 'gray', clim=(0.0, 1.0))
    c = plt.Circle((w - 1, w - 1), r, color=(0.9, 0.9, 0, 0.5), linewidth=1, fill=False)
    ax[1].add_patch(c)

    # ## Original for QC
    # ax[1].set_title('Native Contrast\nblob_detection radius:{}'.format(r))
    # ax[1].imshow(image, 'gray', clim=(0.0, 1.0))
    # c = plt.Circle((w - 1, w - 1), r, color=(0.9, 0.9, 0, 0.5), linewidth=1, fill=False)
    # ax[1].add_patch(c)

    # ## Equalized for QC
    # ax[2].set_title('Equalized\narea (pixels):{}'.format(int(area)))
    # ax[2].imshow(equalized, 'gray', clim=(0.0, 1.0))
    # c = plt.Circle((w - 1, w - 1), r, color=(0.9, 0.9, 0, 0.5), linewidth=1, fill=False)
    # ax[2].add_patch(c)



    plt.tight_layout()
    if fname:
        plt.savefig(fname+".png")
    else:
        plt.show()
    fig.canvas.draw()

    return True


def plot_flat_crops(flat_crops, blob_extention_ratio=1, blob_extention_radius=0, fname=None):
    '''
    input: flat_crops
    task: call plot_flat_crop many times
    '''
    for i, flat_crop in enumerate(flat_crops):
        if fname:
            plot_flat_crop(flat_crop, blob_extention_ratio=blob_extention_ratio, blob_extention_radius=blob_extention_radius, 
                fname=fname+'.rnd'+str(i))
        else:
            plot_flat_crop(flat_crop, blob_extention_ratio=blob_extention_ratio, blob_extention_radius=blob_extention_radius)


def show_rand_crops(crops, label_filter="na", num_shown=5, 
    blob_extention_ratio=1, blob_extention_radius=0, 
     seed = None, fname=None):
    '''
    crops: the blob crops
    label_filter: 0, 1, -1; "na" means no filter
    fname: None, plot.show(); if fname provided, saved to png
    '''
    from .misc import sub_sample

    if (label_filter != 'na'):
        filtered_idx = [str(int(x)) == str(label_filter) for x in crops[:, 3]]
        crops = crops[filtered_idx, :]

    if len(crops) = 0:
        print('num_blobs after filtering is 0')
        return False

    if (len(crops) >= num_shown):
        print("Samples of {} blobs will be plotted".format(num_shown))
        crops = sub_sample(crops, num_shown, seed)
    else:
        print("all {} blobs will be plotted".format(len(crops)))

    plot_flat_crops(crops,
        blob_extention_ratio=blob_extention_ratio, blob_extention_radius=blob_extention_radius, fname=fname)
    images, labels, rs = parse_blobs(crops)
    [area_calculation(image, r=rs[ind], plotting=True) for ind, image in enumerate(images)]

    return (True)








def pop_label_flat_crops(flat_crops, random=True, seed=1, skip_labeled=True):
    '''
    input: flat_crops
    task: plot padded crop and hard-masked crop side-by-side, and let user label them
    labels: -1 not-labeled, 0 NO, 1 YES
    output: labeled array in the original order
    '''
    print("Input: there are {} blobs unlabeled in {} blobs\n\n".format(sum(flat_crops[:, 3] == -1), len(flat_crops)))

    N = len(flat_crops)
    if random:
        np.random.seed(seed=seed)
        idx = np.random.permutation(N)
        np.random.seed()
    else:
        idx = np.arange(N)

    if skip_labeled:
        idx = idx[flat_crops[idx, 3] == -1]  # only keep unlabeled (-1)

    num_unlabeled = sum(flat_crops[:, 3] == -1)

    i = -1
    while i < len(idx):
        i += 1
        if i >= len(idx):
            break

        plot_flat_crop(flat_crops[idx[i], :])

        label = input('''labeling for the {}/{} unlabeled blob, 
yes=1, no=0, undistinguishable=3, skip=s, go-back=b, excape(pause)=e: '''.format(i + 1, num_unlabeled))

        if label == '1':
            flat_crops[idx[i], 3] = 1  # yes
        elif label == '0':
            flat_crops[idx[i], 3] = 0  # no
        elif label == '3':
            flat_crops[idx[i], 3] = -2  # undistinguishable
        elif label == 's':
            pass
        elif label == 'b':
            i -= 2
        elif label == 'e':
            label = input('are you sure to quit?(y/n)')
            if label == 'y':
                print("there are {} blobs unlabeled\n\n".format(sum(flat_crops[:, 3] == -1)))
                print("labeling stopped manually")
                break
            else:
                print('continued')
                i -= 1
        else:
            print('invalid input, please try again')
            i -= 1

        print('new label: ', label, flat_crops[idx[i], 0:4])
        print("there are {} blobs unlabeled\n\n".format(sum(flat_crops[:, 3] == -1)))
        clear_output()

    return flat_crops


