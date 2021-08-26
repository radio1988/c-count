from ccount.img.equalize import equalize, block_equalize
from ccount.img.transform import down_scale, pad_with




import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path
import re
import time
import tracemalloc
import gc



from math import sqrt
from skimage import data, img_as_float
from skimage.draw import circle
from skimage import exposure
from skimage.color import rgb2gray
from skimage.transform import rescale, resize, downscale_local_mean
from IPython.display import clear_output
from random import randint
from time import sleep


## Blob related ## 
def vis_blob_on_block(blobs, block_img_equ, block_img_ori, 
    blob_extention_ratio=1.0, blob_extention_radius=0, scaling = 4, fname=None):
    '''
    blobs: blob info array [n, 0:3]
    block_img_equ: corresponding block_img equalized
    block_img_ori: block_img before equalization
    plot: plot block_img with blobs in yellow circles
    '''
    print('scaling of visualization is ', scaling)
    blobs = blobs[:, 0:3]
    blobs = blobs/scaling
    block_img_equ = down_scale(block_img_equ, scaling)
    block_img_ori = down_scale(block_img_ori, scaling)
    

    fig, axes = plt.subplots(2, 1, figsize=(80, 40), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].set_title('Equalized Image')
    ax[0].imshow(block_img_equ, 'gray', interpolation='nearest', clim=(0.0, 1.0))
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), 
                       r * blob_extention_ratio + blob_extention_radius, 
                       color=(0.9, 0.9, 0, 0.5), linewidth=1,
                       fill=False)  # r*1.3 to get whole blob
        ax[0].add_patch(c)
    # ax[0].set_axis_off()
    ax[1].set_title("Original Image")
    ax[1].imshow(block_img_ori, 'gray', interpolation='nearest')
    for blob in blobs:
        y, x, r = blob
        d = plt.Circle((x, y), 
                       r * blob_extention_ratio + blob_extention_radius, 
                       color=(0.9, 0.9, 0, 0.5), linewidth=1,
                       fill=False)  # r*1.3 to get whole blob
        ax[1].add_patch(d)
    # ax[0].set_axis_off()
    plt.tight_layout()
    if fname:
        plt.savefig(fname)
    else:
        plt.show()
        




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




def plot_flat_crop(flat_crop, blob_extention_ratio=1, blob_extention_radius=0, fname=None, plot_area=True):
    '''
    input: one padded crop of a blob
    plt: yellow circle and hard-masked side-by-side
    columns: [y, x, r_, area, place_holder, flattened cropped_blob_img (crop_width^2)]
    return: cropped image and hard-masked image
    '''
    # reshape
    [y, x, r, label, area, place_holder] = flat_crop[0:6]
    r_ = r * blob_extention_ratio + blob_extention_radius

    flat = flat_crop[6:]
    w = int(sqrt(len(flat)) / 2)
    image = np.reshape(flat, (w + w, w + w))
    image = (image - np.min(image)) / np.max(image)
    #print("max_pixel value:", round(np.max(image), 3))

    # Equalized
    equalized = equalize(image)

    # hard mask creating training data
    # hard_masked = mask_image(equalized, r=r_)

    area_plot = area_calculation(image, r, plotting=True)

    fig, axes = plt.subplots(1, 4, figsize=(8, 32), sharex=False, sharey=False)
    ax = axes.ravel()
    ## Auto Contrast For labeler
    ax[0].set_title('For Labeling\ncurrent label:{}'.format(int(label)))
    ax[0].imshow(image, 'gray')
    c = plt.Circle((w - 1, w - 1), r_, color=(0.9, 0.9, 0, 0.5), linewidth=1, fill=False)
    ax[0].add_patch(c)

    ## Original for QC
    ax[1].set_title('Native Contrast\nblob_detection radius:{}'.format(r))
    ax[1].imshow(image, 'gray', clim=(0.0, 1.0))
    c = plt.Circle((w - 1, w - 1), r_, color=(0.9, 0.9, 0, 0.5), linewidth=1, fill=False)
    ax[1].add_patch(c)

    ## Equalized for QC
    ax[2].set_title('Equalized\narea (pixels):{}'.format(int(area)))
    ax[2].imshow(equalized, 'gray', clim=(0.0, 1.0))
    c = plt.Circle((w - 1, w - 1), r_, color=(0.9, 0.9, 0, 0.5), linewidth=1, fill=False)
    ax[2].add_patch(c)

    ## area calculation
    ax[3].set_title('Area Calculation\narea (pixels):{}'.format(int(area)))
    ax[3].imshow(area_plot, 'gray', clim=(0.0, 1.0))
    c = plt.Circle((w - 1, w - 1), r_, color=(0.9, 0.9, 0, 0.5), linewidth=1, fill=False)
    ax[3].add_patch(c)

    plt.tight_layout()
    if fname:
        plt.savefig(fname+".png")
    else:
        plt.show()
    fig.canvas.draw()

    return True

def plot_flat_crops(flat_crops, blob_extention_ratio=1, blob_extention_radius=0, fname=None, plot_area=False):
    '''
    input: flat_crops
    task: plot padded crop and hard-masked crop side-by-side
    '''
    for i, flat_crop in enumerate(flat_crops):
        if fname:
            plot_flat_crop(flat_crop, blob_extention_ratio=blob_extention_ratio, blob_extention_radius=blob_extention_radius, 
                fname=fname+'.rnd'+str(i), plot_area=plot_area)
        else:
            plot_flat_crop(flat_crop, blob_extention_ratio=blob_extention_ratio, blob_extention_radius=blob_extention_radius,
                plot_area=plot_area)




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


def show_rand_crops(crops, label_filter="na", num_shown=5, 
    blob_extention_ratio=1, blob_extention_radius=0, 
    plot_area=False, seed = None, fname=None):
    '''
    blobs: the blobs crops
    label_filter: 0, 1, -1; "na" means no filter
    fname: None, plot.show(); if fname provided, saved to png
    '''
    if (label_filter != 'na'):
        filtered_idx = [str(int(x)) == str(label_filter) for x in crops[:, 3]]
        #print('labels:',[str(int(x)) for x in crops[:, 3]])
        #print('filter:', filtered_idx)
        crops = crops[filtered_idx, :]


    if (len(crops) >= num_shown):
        print(num_shown, "blobs will be plotted")
        if seed:
            np.random.seed(seed)
        randidx = np.random.choice(range(len(crops)), num_shown, replace=False)
        np.random.seed()

        if (plot_area):
            Images, Labels, Rs = parse_blobs(crops[randidx, :])
            [area_calculation(image, r=Rs[ind], plotting=True) for ind, image in enumerate(Images)]

        plot_flat_crops(crops[randidx, :], 
            blob_extention_ratio=blob_extention_ratio, blob_extention_radius=blob_extention_radius, fname=fname)
    elif (len(crops) > 0):
        print("Only", len(crops), 'blobs exist, and will be plotted')
        plot_flat_crops(crops,
            blob_extention_ratio=blob_extention_ratio, blob_extention_radius=blob_extention_radius, fname=fname)

        if (plot_area):
            Images, Labels, Rs = parse_blobs(crops)
            [area_calculation(image, r=Rs[ind], plotting=True) for ind, image in enumerate(Images)]
    else:
        print('num_blobs after filtering is 0')
        
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



    


def remove_edge_crops(flat_blobs):
    """
    some crops of blobs contain edges, because they are from the edge of scanned areas or on the edge of the well
    use this function to remove blobs with obvious long straight black/white lines
    """
    import cv2
    good_flats = []
    for i in range(0, flat_blobs.shape[0]):
        flat = flat_blobs[i,]
        crop = reshape_img_from_flat(flat)
        crop = crop * 255
        crop = crop.astype(np.uint8)
    
        crop = cv2.blur(crop,(4,4))
    
        edges = cv2.Canny(crop,50,150,apertureSize = 3)

        minLineLength = 40
        maxLineGap = 10
        lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength,maxLineGap)
    
        if lines is not None: # has lines
            pass
#             print(lines.shape)
#             for i in range(0, lines.shape[0]):
#                 for x1,y1,x2,y2 in lines[i]:
#                     cv2.line(edges,(x1,y1),(x2,y2),(255,255,0, 0.8),6)
#             plt.title("Bad")
#             plt.imshow(crop)
#             plt.show()
        else: # no lines
            good_flats.append(flat)
#             plt.title(str(i))
#             plt.imshow(crop, 'gray')
#             plt.show()
    #         plt.imshow(edges, "gray")
    #         plt.title(str(i))
    #         plt.show()
    #         print("Good")
    
    good_flats = np.stack(good_flats)
    return (good_flats)





# TEST SCALING and Equalization
# i = 0; j = 0; l = 2048
# block = image[2048*i : 2048*(i+1), 
#               2048*j : 2048*(j+1)]
# block_equ = equalize(block)
# block_equ_small = down_scale(block_equ, 2)

# print('block image: ', block.shape)
# print ("resized image: ", block_equ_small.shape)

# fig, axes = plt.subplots(1, 3, figsize=(20, 60), sharex=False, sharey=False)
# ax = axes.ravel()show_rand_crops
# ax[0].set_title('input block')
# ax[0].imshow(block, 'gray')
# ax[1].set_title('equalized block')
# ax[1].imshow(block_equ, 'gray')
# ax[2].set_title('scaled block')
# ax[2].imshow(block_equ_small, 'gray')

def split_train_valid(array, training_ratio):
    """
    Split into train and valid
    :param array: 2D array, each row is a sample
    :param ratio: ratio of train in all, e.g. 0.7
    :return: two arrays
    """
    N = array.shape[0]
    N1 = int(N * training_ratio)
    np.random.seed(3)
    np.random.shuffle(array)
    np.random.seed()
    train = array[0:N1]
    valid = array[N1:]
    return train, valid




def balancing_by_removing_no(blobs):
    '''
    balance yes/no ratio to 1, by removing excess NO samples
    :return: balanced blobs (with less samples)
    '''
    print('Before balancing:')
    blobs_stat(blobs)

    idx_yes = np.arange(0, blobs.shape[0])[blobs[:, 3] == 1]
    idx_no = np.arange(0, blobs.shape[0])[blobs[:, 3] == 0]
    N_Yes = len(idx_yes)
    N_No = len(idx_no)

    if N_No > N_Yes:
        print('number of No matched to yes by sub-sampling')
        idx_no = np.random.choice(idx_no, N_Yes, replace=False)
        idx_choice = np.concatenate([idx_yes, idx_no])
        np.random.seed(2)
        np.random.shuffle(idx_choice)
        np.random.seed()
        blobs = blobs[idx_choice,]

    print("After balancing by removing neg samples")
    blobs_stat(blobs)

    return blobs


def balancing_by_duplicating(blobs):
    '''
    balance yes/no ratio to 1, by duplicating blobs in the under-represented group
    only yes/no considered
    undistinguishable not altered
    result randomized to avoid problems in training
    :return: balanced blobs (with less samples)
    '''
    print('Before balancing:')
    blobs_stat(blobs)

    idx_yes = np.arange(0, blobs.shape[0])[blobs[:, 3] == 1]
    idx_no = np.arange(0, blobs.shape[0])[blobs[:, 3] == 0]
    idx_unsure = np.arange(0, blobs.shape[0])[blobs[:, 3] == -2]
    N_Yes = len(idx_yes)
    N_No = len(idx_no)
    N_unsure = len(idx_unsure)

    # todo: include unsure
    if N_No > N_Yes:
        print('number of No matched to Yes by re-sampling')
        idx_yes = np.random.choice(idx_yes, N_No, replace=True)  # todo: some yes data lost when N_No small
    elif N_Yes > N_No:
        print('number of Yes matched to No by re-sampling')
        idx_no = np.random.choice(idx_no, N_Yes, replace=True)
    idx_choice = np.concatenate([idx_yes, idx_no, idx_unsure])  # 3 classes
    np.random.shuffle(idx_choice)
    blobs = blobs[idx_choice, ]

    print("After balancing by adding positive samples")
    blobs_stat(blobs)

    return blobs





def augment_images(Images, aug_sample_size):
    '''
    Input images (n_samples, 2*w, 2*w)
    Process: Augmentation; Normalization back to [0, 1]
    Output augmented images of the same shape
    :param Images:
    :return: augImages
    '''
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    import imgaug as ia
    from imgaug import augmenters as iaa


    sometimes = lambda aug: iaa.Sometimes(0.9, aug)

    w2 = Images.shape[1]
    Images = Images.reshape(len(Images), w2, w2, 1) # formatting

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.5), # vertically flip 20% of all images
            sometimes(iaa.Affine(
                # todo: more strict; no scaling down
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                rotate=(-90, 90), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 1), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
        ],
        random_order=True
    )

    Images_ = seq.augment_images(Images)
    while Images_.shape[0] < aug_sample_size:
        _ = seq.augment_images(Images)
        Images_ = np.vstack((Images_, _))
    Images = Images_[0:aug_sample_size, :]
    print('shape:', Images.shape, 'after augment_images')

    Images = Images.reshape(len(Images), w2, w2)  # formatting back
    Images = np.array([float_image_auto_contrast(image) for image in Images])
    return Images


def F1(y_pred, y_true):
    from keras import backend as K
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    f1 = 2*((precision*recall)/(precision+recall+K.epsilon()))
    return f1

def f1_score(precision, recall):
    return 2*(precision*recall/(precision+recall+1e-07))

def F1_calculation(predictions, labels):
    print("F1_calculation for sure labels only")
    idx = (labels == 1) | (labels == 0)  # sure only
    labels = labels[idx, ]
    predictions = predictions[idx, ]


    TP = np.sum(np.round(predictions * labels))
    PP = np.sum(np.round(labels))
    recall = TP / (PP + 1e-7)

    PP2 = np.sum(np.round(predictions))
    precision = TP/(PP2 + 1e-7)

    F1 = 2*((precision*recall)/(precision+recall+1e-7))

    print('Precition: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(precision*100, recall*100, F1*100))

    return F1


def preprocessing_imgs(Images, Rs, Labels, scaling_factor):
    # Downscale images (todo: downscale as the first step)
    print("Downscaling images by ", scaling_factor)
    Images = np.array([down_scale(image, scaling_factor=scaling_factor) for image in Images])
    ## Downscale w and R
    print('w after scaling:', w)
    Rs = Rs/scaling_factor

    # Equalize images (todo: test equalization -> scaling)
    # todo: more channels (scaled + equalized + original)
    print("Equalizing images...")
    # todo:  Possible precision loss when converting from float64 to uint16
    Images = np.array([equalize(image) for image in Images])

    # Mask images
    print("Masking images...")
    Images = np.array([mask_image(image, r=Rs[ind]) for ind, image in enumerate(Images)])

    # Normalizing images
    print("Normalizing images...")
    # todo:  Possible precision loss when converting from float64 to uint16
    Images = np.array([float_image_auto_contrast(image) for image in Images])

    return Images, Rs, Labels, w


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE  # single core
#from MulticoreTSNE import MulticoreTSNE as TSNE  # MCORE

def cluster_scatterplot(df2d, labels, title):
    '''
    PCA or t-SNE 2D visualization

    `cluster_scatterplot(tsne_projection, cluster_info.Cluster.values.astype(int),
                    title='projection.csv t-SNE')`

    :param df2d: PCA or t-SNE projection df, cell as row, feature as columns
    :param labels:
    :param title:
    :return:
    '''
    legends = np.unique(labels)
    print('all labels:', legends)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)

    for i in legends:
        _ = df2d.iloc[labels == i]
        num_blobs = str(len(_))
        percent_cells = str(round(int(num_blobs) / len(df2d) * 100, 1)) + '%'
        ax.scatter(_.iloc[:, 0], _.iloc[:, 1],
                   alpha=0.5, marker='.',
                   label='c' + str(i) + ':' + num_blobs + ', ' + percent_cells
                   )

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.xlabel('legend format:  cluster_id:num-cells')

    #plt.savefig(title + '.png', bbox_inches='tight')
    plt.show()
    plt.close('all')


def pca_tsne(df_gene_col, cluster_info=None, title='data', 
             #dir='plots',
             num_pc=50, num_tsne=2, ncores=8):
    '''
    PCA and tSNE plots for DF_cell_row, save projections.csv
    :param df_cell_row: data matrix, features as columns, e.g. [cell, gene]
    :param cluster_info: cluster_id for each cell_id
    :param title: figure title, e.g. Late
    :param num_pc: 50
    :param num_tsne: 2
    :return: tsne_df, plots saved, pc_projection.csv, tsne_projection.csv saved
    '''

#     if not os.path.exists(dir):
#         os.makedirs(dir)

#     title = './' + dir + '/' + title

    df = df_gene_col
    if cluster_info is None:
        cluster_info = pd.DataFrame(0, index=df.index, columns=['cluster_id'])

    tic = time.time()
    # PCA
    pca = PCA(n_components=num_pc)
    pc_x = pca.fit_transform(df)
    df_pc_df = pd.DataFrame(data=pc_x, index=df.index, columns=range(num_pc))
    df_pc_df.index.name = 'cell_id'
    df_pc_df.columns.name = 'PC'
    #df_pc_df.to_csv(title + '.pca.csv')
    print('dim before PCA', df.shape)
    print('dim after PCA', df_pc_df.shape)
    print('explained variance ratio: {}'.format(
        sum(pca.explained_variance_ratio_)))

    colors = cluster_info.reindex(df_pc_df.index)
    colors = colors.dropna().iloc[:, 0]
    print('matched cluster_info:', colors.shape)
    print('unmatched data will be excluded from the plot')  # todo: include unmatched

    df_pc_ = df_pc_df.reindex(colors.index)  # only plot labeled data?
    cluster_scatterplot(df_pc_, colors.values.astype(str), title=title + ' (PCA)')

#     # tSNE
#     print('MCORE-TSNE, with ', ncores, ' cores')
#     df_tsne = TSNE(n_components=num_tsne, n_jobs=ncores).fit_transform(df_pc_)
#     print('tsne done')
#     df_tsne_df = pd.DataFrame(data=df_tsne, index=df_pc_.index)
#     print('wait to output tsne')
#     df_tsne_df.to_csv(title + '.tsne.csv')
#     print('wrote tsne to output')
#     cluster_scatterplot(df_tsne_df, colors.values.astype(str), title=title + ' ('
#                                                                              't-SNE)')
    toc = time.time()
    print('took {:.1f} seconds\n'.format(toc - tic))

    return df_pc_df



