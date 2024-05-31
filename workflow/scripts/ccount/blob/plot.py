import matplotlib.pyplot as plt
import numpy as np
from ..blob.misc import parse_crops
from ..img.equalize import equalize

def flat2image(flat_crop):
    from math import sqrt
    import numpy as np
    flat = flat_crop[6:]
    w = int(sqrt(len(flat)) / 2)
    image = np.reshape(flat, (w + w, w + w))
    return image


def visualize_blob_detection(image, blob_locs, 
    blob_extention_ratio=1.4, blob_extention_radius=10, fname=None):
    '''
    image: image where blobs were detected from
    blob_locs: blob info array n x 4 [x, y, r, label], crops also works, only first 3 columns used
    blob_locs: if labels in the fourth column provided, use that to give colors to blobs

    output: image with yellow circles around blobs
    '''
    from ..img.transform import down_scale
    px = 1/plt.rcParams['figure.dpi']

    # blob_locs[:, 0:3] = blob_locs[:, 0:3]/scaling
    #image = down_scale(image, scaling)
    print("image shape:", image.shape)

    fig, ax = plt.subplots(figsize=(image.shape[1]*px+0.5, image.shape[0]*px+0.5))
    ax.imshow(image, 'gray')

    print("blob shape:", blob_locs.shape)
    if blob_locs.shape[1] >= 4:
        ax.set_title('Visualizing blobs:\n\
            Red: Yes, Blue: No, Green: Others')
        labels = blob_locs[:,3]
        red = labels == 1
        blue = labels == 0
        green = [x not in [0, 1] for x in labels]
        for loc in blob_locs[red,0:3]:
            y, x, r = loc
            RED = plt.Circle((x, y), 
                           r * blob_extention_ratio + blob_extention_radius, 
                           color=(1, 0, 0, 0.7), linewidth=2,
                           fill=False) 
            ax.add_patch(RED)

        for loc in blob_locs[blue,0:3]:
            y, x, r = loc
            BLUE = plt.Circle((x, y), 
                           r * blob_extention_ratio + blob_extention_radius, 
                           color=(0, 0, 1, 0.7), linewidth=2,
                           fill=False) 
            ax.add_patch(BLUE)

        for loc in blob_locs[green,0:3]:
            y, x, r = loc
            GREEN = plt.Circle((x, y), 
                           r * blob_extention_ratio + blob_extention_radius, 
                           color=(0, 1, 0, 0.5), linewidth=2,
                           fill=False) 
            ax.add_patch(GREEN)
    else:
        print('no label provided')
        for loc in blob_locs[:, 0:3]:
            ax.set_title('Visualizing blobs')
            y, x, r = loc
            YELLOW = plt.Circle((x, y), 
                           r * blob_extention_ratio + blob_extention_radius, 
                           color=(0.9, 0.9, 0, 0.5), linewidth=1,
                           fill=False) 
            ax.add_patch(YELLOW)

    if fname:
        plt.savefig(fname)
    else:
        plt.show()


def visualize_blob_compare(image, blob_locs, blob_locs2, 
    blob_extention_ratio=1.4, blob_extention_radius=10, fname=None):
    '''
    image: image where blobs were detected from
    blob_locs: blob info array n x 4 [x, y, r, label]
    blob_locs2: ground truth

    output: image with colored circles around blobs
        GT, Label
        0, 0, blue
        1, 1, pink
        0, 1, red
        1, 0, purple
    '''
    from ..img.transform import down_scale
    from ccount.clas.metrics import F1_calculation
    px = 1/plt.rcParams['figure.dpi']

    # blob_locs[:, 0:3] = blob_locs[:, 0:3]/scaling
    #image = down_scale(image, scaling)
    print("image shape:", image.shape)
    print("blob shape:", blob_locs.shape, blob_locs2.shape)

    if not blob_locs2.shape[1] == blob_locs.shape[1]:
        raise ValueError('num of locs in crops and crops2 different')

    if blob_locs.shape[1] <= 3 or blob_locs2.shape[1] <= 3 :
        raise ValueError('crop or crops2 has no label')

    labels = blob_locs[:,3]
    labels2 = blob_locs2[:,3]
    precision, recall, F1 = F1_calculation(labels, labels2)

    fig, ax = plt.subplots(figsize=(image.shape[1]*px+0.5, image.shape[0]*px+0.5))
    ax.imshow(image, 'gray')

    ax.set_title('Visualizing blobs:\n\
        Red: FP, Yellow: FN, Green: TP, Blue: TN\n\
        Precision: {:.3f}, Recall: {:.3f}, F1: , {:.3f}'.format(precision, recall, F1))

    fp = [gt == 0 and clas == 1 for gt, clas in zip(labels2, labels)]
    fn = [gt == 1 and clas == 0 for gt, clas in zip(labels2, labels)]
    tp = [gt == 1 and clas == 1 for gt, clas in zip(labels2, labels)]
    tn = [gt == 0 and clas == 0 for gt, clas in zip(labels2, labels)]

    for loc in blob_locs[fp,0:3]:
        y, x, r = loc
        FP = plt.Circle((x, y), 
                       r * blob_extention_ratio + blob_extention_radius, 
                       color=(1, 0, 0, 0.7), linewidth=2,
                       fill=False) 
        ax.add_patch(FP)

    for loc in blob_locs[fn,0:3]:
        y, x, r = loc
        FN = plt.Circle((x, y), 
                       r * blob_extention_ratio + blob_extention_radius, 
                       color=(1, 1, 0, 0.7), linewidth=2,
                       fill=False) 
        ax.add_patch(FN)

    for loc in blob_locs[tp,0:3]:
        y, x, r = loc
        TP = plt.Circle((x, y), 
                       r * blob_extention_ratio + blob_extention_radius, 
                       color=(0, 1, 0, 0.7), linewidth=2,
                       fill=False) 
        ax.add_patch(TP)


    for loc in blob_locs[tn,0:3]:
        y, x, r = loc
        TN = plt.Circle((x, y), 
                       r * blob_extention_ratio + blob_extention_radius, 
                       color=(0, 0, 1, 0.7), linewidth=2,
                       fill=False) 
        ax.add_patch(TN)


    if fname:
        plt.savefig(fname)
    else:
        plt.show()

def plot_flat_crop(flat_crop, blob_extention_ratio=1.4, blob_extention_radius=10, 
    image_scale=1, fname=None, equalization=False):
    '''
    input: flat_crop of a blob, e.g. (160006,)
    output: two plots
        - left: original image with yellow circle
        - right: binary for area calculation
    '''
    from math import sqrt
    from ..img.auto_contrast import float_image_auto_contrast
    import matplotlib.pyplot as plt
    import numpy as np

    if len(flat_crop) >= 6:
        [y, x, r, label, area, place_holder] = flat_crop[0:6]
    else: 
        [y, x, r] = flat_crop[0:3]
        label = 5
        area = 0

    r = r * blob_extention_ratio + blob_extention_radius
    image = flat2image(flat_crop)
    image = float_image_auto_contrast(image)
    w = sqrt(len(flat_crop) - 6) / 2
    W = w * image_scale / 30
    area = flat_crop[6]

    if equalization:
        image = equalize(image)

    fig, ax = plt.subplots(figsize=(W, W))
    ax.set_title('Image for Labeling\ncurrent label:{}\n\
        x:{}, y:{}, r:{}, area:{}'.format(int(label), x ,y, r, area))
    ax.imshow(image, 'gray')
    c = plt.Circle((w, w), r , 
                   color=(1, 1, 0, 0.7), linewidth=2,
                   fill=False) 
    ax.add_patch(c)

    if fname:
        plt.savefig(fname)
    else:
        plt.show()

    return image


def plot_flat_crops(crops, blob_extention_ratio=1, blob_extention_radius=0, fname=None):
    '''
    input: crops
    task: call plot_flat_crop many times
    '''
    for i, flat_crop in enumerate(crops):
        if fname:
            plot_flat_crop(flat_crop, 
                blob_extention_ratio=blob_extention_ratio, 
                blob_extention_radius=blob_extention_radius, 
                fname=fname+'.rnd'+str(i)+'.jpg')
        else:
            plot_flat_crop(flat_crop, 
                blob_extention_ratio=blob_extention_ratio, 
                blob_extention_radius=blob_extention_radius)



    

def show_rand_crops(crops, label_filter="na", num_shown=1, 
    blob_extention_ratio=1, blob_extention_radius=0, seed = None, fname=None):
    '''
    crops: the blob crops
    label_filter: 0, 1, 5; "na" means no filter
    fname: None, plot.show(); if fname provided, saved to png
    '''
    from .misc import sub_sample    

    if (label_filter != 'na'):
        filtered_idx = [str(int(x)) == str(label_filter) for x in crops[:, 3]]
        crops = crops[filtered_idx, :]

    if len(crops) == 0:
        print('num_blobs after filtering is 0')
        return False

    if (len(crops) >= num_shown):
        print("Samples of {} blobs will be plotted".format(num_shown))
        crops = sub_sample(crops, num_shown, seed)
    else:
        print("all {} blobs will be plotted".format(len(crops)))

    plot_flat_crops(
        crops,
        blob_extention_ratio=blob_extention_ratio, 
        blob_extention_radius=blob_extention_radius, 
        fname=fname)

    images, labels, rs = parse_crops(crops)

    return (True)


def pop_label_flat_crops(crops, random=True, seed=1, skip_labels=[0,1,2,3]):
    '''
    input: 
        crops
    task: 
        plot padded crop, let user label them
    labels: 
        no: 0, yes: 1, groupB: 2, uncertain: 3, artifacts: 4, unlabeled: 5 
        never use neg values
    skipLablels:
        crops with current labels in these will be skipped, to save time
    output: 
        labeled array in the original order
    '''
    from IPython.display import clear_output


    N = len(crops)
    if random:
        np.random.seed(seed=seed)
        idx = np.random.permutation(N)
        np.random.seed()
    else:
        idx = np.arange(N)

    labels = crops[idx, 3]
    keep = [x not in skip_labels for x in labels]
    idx = idx[keep] 

    num_to_label = sum(keep)
    print("Input: there are {} blobs to label in {} blobs".\
        format(num_to_label, len(crops)))

    i = -1
    while i < len(idx):
        i += 1
        if i >= len(idx):
            break

        plot_flat_crop(crops[idx[i], :])

        label = input('''labeling for the {}/{} blob, \
            yes=1, no=0, skip=s, go-back=b, excape(pause)=e'''.\
            format(i + 1, num_to_label))

        if label == '1':
            crops[idx[i], 3] = 1 
        elif label == '0':
            crops[idx[i], 3] = 0
        elif label == '2':
            crops[idx[i], 3] = 2
        elif label == '3':
            crops[idx[i], 3] = 3
        elif label == 's':
            pass
        elif label == 'b':
            i -= 2
        elif label == 'e':
            label = input('are you sure to quit?(y/n)')
            if label == 'y':
                print("labeling stopped manually")
                break
            else:
                print('continued')
                i -= 1
        else:
            print('invalid input, please try again')
            i -= 1

        print('new label: ', label, crops[idx[i], 0:4])
        clear_output()

    return crops


