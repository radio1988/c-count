import copy

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from ccount.img.auto_contrast import float_image_auto_contrast
from ..blob.misc import get_blob_statistics, parse_crops, crop_width



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
    sometimes = lambda aug: iaa.Sometimes(0.9, aug)

    w2 = images.shape[1]
    images = images.reshape(len(images), w2, w2, 1)  # formatting

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.5),  # vertically flip 50% of all images
            sometimes(iaa.Affine(
                # todo: more strict; no scaling down
                scale=(1, 1.2),  # larger range hoping to eliminate size classifier
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                rotate=(-360, 360),
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 1),  # todo: [0, 255] for uint8 images
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
        ],
        random_order=True
    )

    images_ = images.copy()
    while images_.shape[0] < aug_sample_size:
        _ = seq.augment_images(images)
        images_ = np.vstack((images_, _))
    images = images_[0:aug_sample_size, :]
    print('shape:', images.shape, 'after augment_images')

    images = images.reshape(len(images), w2, w2)  # formatting back
    images = np.array([float_image_auto_contrast(image) for image in images])
    return images


def augment_crops(images, labels, Rs, aug_sample_size):
    '''
    Input: crops (images, labels, Rs)
    Process:
    - if n_crops < aug_sample_size, augmentation performed so n_crops == aug_sample_size
    - if n_crops >= aug_sample_size, no augmentation will be performed
    return: augmented crops (images, labels, Rs)
    '''
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases
    # e.g. ·
    sometimes = lambda aug: iaa.Sometimes(0.9, aug)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.5),  # vertically flip 50% of all images
            sometimes(iaa.Affine(
                # todo: more strict; no scaling down
                scale=(1, 1.2),  # larger range hoping to eliminate size classifier
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                rotate=(-360, 360),
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 1),  # todo: [0, 255] for uint8 images
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
        ],
        random_order=True
    )

    w = images.shape[1]
    images = images.reshape(len(images), w, w, 1)  # formatting

    images_ = images.copy()
    labels_ = copy.deepcopy(labels)
    Rs_ = copy.deepcopy(Rs)

    while images_.shape[0] < aug_sample_size:
        _ = seq.augment_images(images)
        images_ = np.vstack((images_, _))
        labels_ = np.concatenate((labels_, labels))
        Rs_ = np.concatenate((Rs_, Rs))
    images = images_[0:aug_sample_size, :]
    labels = labels_[0:aug_sample_size]
    Rs = Rs_[0:aug_sample_size]
    images = images.reshape(len(images), w, w)  # formatting back
    images = np.array([float_image_auto_contrast(image) for image in images])
    return images, labels, Rs


import numpy as np
from ..blob.misc import get_blob_statistics, parse_crops
from ..clas.augment_images import augment_images


def balance_by_removal(blobs):
    '''
    balance yes/no ratio to 1, by removing excess NO samples
    :return: balanced blobs (with less samples)
    '''
    print('Before balancing:')
    get_blob_statistics(blobs)

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
    get_blob_statistics(blobs)

    return blobs


def balance_by_duplication(blobs, maxN=160000):
    '''
    balance yes/no ratio to 1, by duplicating blobs in the under-represented group
    only yes/no considered
    undistinguishable not altered
    result randomized to avoid problems in training
    input: crops
    :return: balanced blobs (with less samples)
    '''
    print('Before balancing:')
    get_blob_statistics(blobs)

    idx_yes = np.arange(0, blobs.shape[0])[blobs[:, 3] == 1]
    idx_no = np.arange(0, blobs.shape[0])[blobs[:, 3] == 0]
    idx_unsure = np.arange(0, blobs.shape[0])[blobs[:, 3] == -2]
    N_Yes = len(idx_yes)
    N_No = len(idx_no)
    # N_unsure = len(idx_unsure)

    if N_Yes > maxN // 2:
        idx_yes = np.random.choice(idx_yes, maxN // 2, replace=False)
    if N_No > maxN // 2:
        idx_no = np.random.choice(idx_no, maxN // 2, replace=False)
    if N_Yes < maxN // 2:
        idx_yes = np.random.choice(idx_yes, maxN // 2, replace=True)
    if N_No < maxN // 2:
        idx_no = np.random.choice(idx_no, maxN // 2, replace=True)

    yes_images, yes_labels, yes_Rs = parse_crops(blobs[idx_yes,])

    idx_choice = np.concatenate([idx_yes, idx_no])
    np.random.shuffle(idx_choice)
    blobs = blobs[idx_choice, ]

    print("After balancing by adding positive samples")
    get_blob_statistics(blobs)

    return blobs

import numpy as np


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

    print('Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(precision*100, recall*100, F1*100))

    return [precision, recall, F1]

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
    import time
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE  # single core
    #from MulticoreTSNE import MulticoreTSNE as TSNE  # MCORE
    df = df_gene_col
    if cluster_info is None:
        cluster_info = pd.DataFrame(0, index=df.index, columns=['cluster_id'])

    tic = time.time()
    # PCA
    num_pc = min(num_pc, df.shape[0])
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
    import numpy as np
    import matplotlib.pyplot as plt

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



def split_data(array, training_ratio):
    """
    Split into train and valid
    seed is always 3
    :param array: 2D array, each row is a sample
    :param ratio: ratio of train in all, e.g. 0.7
    :return: two arrays
    """
    import numpy as np
    N = array.shape[0]
    N1 = int(N * training_ratio)
    np.random.seed(3)
    np.random.shuffle(array)
    np.random.seed()
    train = array[0:N1]
    valid = array[N1:]
    return train, valid

