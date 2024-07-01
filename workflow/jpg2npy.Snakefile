import os


"""
Dir input assumptions:
- data/czi
- data/label_img: the sample names can't have dot `.` in them

output assumptions:
- res/label_locs
- res/label_locs/vis/

"""


INPUT_PATH = 'data/label_img'
IMG_SUFFIX = '.crops.clas.npy.gz'  # point5U_Epo_3-Stitching-11.1.crops.clas.npy.gz


def sampleName_from_sceneName(sceneName):
    """
    assume there is no dot `.` in sample name
    sceneName:  point5U_Epo_3-Stitching-11.1
    sampleName:  point5U_Epo_3-Stitching-11
    """
    sampleName = sceneName.split('.')[0]
    return sampleName


def sceneIndex_from_sceneName(sceneName):
    """
    assume there is no dot `.` in sample name
    sceneName:  point5U_Epo_3-Stitching-11.1
    sceneIndex:  1
    """
    sceneIndex = sceneName.split('.')[1]
    return sceneIndex


def get_jpg_samples(INPUT_PATH):
    filenames = os.listdir(INPUT_PATH)
    jpg_filenames = [filename for filename in filenames if
                     filename.endswith(".jpg")]  # point5U_Epo_3-Stitching-11.1.crops.clas.npy.gz.jpg
    basenames = [os.path.splitext(filename)[0] for filename in
                 jpg_filenames]  # point5U_Epo_3-Stitching-11.1.crops.clas.npy.gz
    scenes = [x.replace(IMG_SUFFIX,"") for x in basenames]  #  point5U_Epo_3-Stitching-11.1
    samples = [sampleName_from_sceneName(x) for x in scenes]  # point5U_Epo_3-Stitching-11
    return samples, scenes


SAMPLES, SCENES = get_jpg_samples(INPUT_PATH)
# print(SAMPLES)
# print(SCENES)

rule targets:
    input:
        label_locs=expand('res/label_locs/{scene}.label.npy.gz',scene=SCENES)



include: 'rules/jpg2locs.smk'  # 900M RAM usage for each job
