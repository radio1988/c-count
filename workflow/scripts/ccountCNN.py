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

    images_ = seq.augment_images(images)
    while images_.shape[0] < aug_sample_size:
        _ = seq.augment_images(images)
        images_ = np.vstack((images_, _))
    images = images_[0:aug_sample_size, :]
    print('shape:', images.shape, 'after augment_images')

    images = images.reshape(len(images), w2, w2)  # formatting back
    images = np.array([float_image_auto_contrast(image) for image in images])
    return images


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


def preprocessing_imgs(images, rs, labels, scaling_factor):
    from ccount.blob.mask_image import mask_image
    # Downscale images (todo: downscale as the first step)
    print("Downscaling images by ", scaling_factor)
    images = np.array([down_scale(image, scaling_factor=scaling_factor) for image in images])
    ## Downscale w and R
    print('w after scaling:', w)
    rs = rs/scaling_factor

    # Equalize images (todo: test equalization -> scaling)
    # todo: more channels (scaled + equalized + original)
    print("Equalizing images...")
    # todo:  Possible precision loss when converting from float64 to uint16
    images = np.array([equalize(image) for image in images])

    # Mask images
    print("Masking images...")
    images = np.array([mask_image(image, r=rs[ind]) for ind, image in enumerate(images)])

    # Normalizing images
    print("Normalizing images...")
    # todo:  Possible precision loss when converting from float64 to uint16
    images = np.array([float_image_auto_contrast(image) for image in images])

    return images, rs, labels, w


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



