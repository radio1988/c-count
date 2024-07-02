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
        label_locs=expand('res/label_locs/{scene}.label.npy.gz',scene=SCENES),
        label_crops=expand('res/label_crops/{scene}.LABEL.npy.gz',scene=SCENES)


rule jpg2locs:
    """
    900M RAM usage on Mac
    """
    input:
        jpg='data/label_img/{sample}.{sceneIndex}' + IMG_SUFFIX + '.jpg',
        czi="data/czi/{sample}.czi",
        blob_locs="res/blob_locs/{sample}.{sceneIndex}.locs.npy.gz"
    output:
        label_locs='res/label_locs/{sample}.{sceneIndex}.label.npy.gz'
    log:
        'res/label_locs/{sample}.{sceneIndex}.log.txt'
    shell:
        "python workflow/scripts/jpg2npy.py {input.jpg} {input.czi} {input.blob_locs} {wildcards.sceneIndex} {output.label_locs} &> {log}"

rule locs2crops:
    input:
        label_locs='res/label_locs/{sample}.{sceneIndex}.label.npy.gz',
        czi="data/czi/{sample}.czi",
    output:
        'res/label_crops/{sample}.{sceneIndex}.LABEL.npy.gz'
    log:
        'res/label_crops/{sample}.{sceneIndex}.LABEL.npy.gz.log.txt'
    shell:
        "python workflow/scripts/blob_cropping.py -czi {input.czi} -locs {input.label_locs} -i {wildcards.sceneIndex} \
        -config config.yaml -o {output} &> {log}"

