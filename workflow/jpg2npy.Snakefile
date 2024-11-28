import os
from scripts.ccount.snake.input_names import input_names


"""
Input assumptions:
- data/czi: raw plate images
    - e.g. 'data/czi/{sample}.czi'
- data/label_img: labeled jpg files.
    - e.g. 'data/label_img/{sample}.{sceneIndex}' + IMG_SUFFIX + '.jpg'
    - all jpg files will be used for sample and scene list collection.
    - sample names should match czi files.
    - no dots `.` in sample names
- config.yaml: blob detection parameters, locs2crops parameters

output assumptions:
- res/blob_locs: can reuse previously detected blob_locs, to save time
- res/crops_locs
- res/label_locs
- res/label_locs/vis/
- res/count.label.csv: count table from labels

Nomenclature:
- sample: plate name
- sceneIndex: there are 4 scenes in a plate with the Merav lab's setup

Logic:
- get samples and sceneIndexes from labeled jpg
- get blob_locs from czi for the samples for the requested sceneIndexes only
- get label_locs and label_crops for the requested sceneIndexes
- aggregate count for them
"""


configfile: "config.yaml"
CZI_PATH = 'data/czi'
LABEL_PATH = 'data/label_img'
IMG_SUFFIX = '.crops.clas.npy.gz'  # point5U_Epo_3-Stitching-11.1.crops.clas.npy.gz


def get_sampleName_from_sceneName(sceneName):
    """
    Input: sceneName (point5U_Epo_3-Stitching-11.1)
    Output: sampleName (point5U_Epo_3-Stitching-11)

    assume there is no dot `.` in sample name
    """
    sampleName = sceneName.split('.')[0]
    return sampleName


def get_sceneIndex_from_sceneName(sceneName):
    """
    Input: sceneName (point5U_Epo_3-Stitching-11.1)
    Output: sceneIndex (1)

    assume there is no dot `.` in sample name
    """
    sceneIndex = sceneName.split('.')[1]
    return sceneIndex


def get_samples_and_sceneIndexes_from_dir(LABEL_PATH):
    """
    Input: LABEL_PATH, e.g. data/label_img/
    Output:
    - SAMPLES: a list of {sample}
    - SCENES: a list of {sample}.{sceneIndex}

    e.g.
    # print(SAMPLES)
    # ['point5U_Epo_1-Stitching-09', 'point5U_Epo_1-Stitching-09', 'point5U_Epo_1-Stitching-09',
    # 'point5U_Epo_1-Stitching-09', '1U_Epo_1-Stitching-01']
    # print(SCENES)
    # ['point5U_Epo_1-Stitching-09.3', 'point5U_Epo_1-Stitching-09.1', 'point5U_Epo_1-Stitching-09.2',
    # 'point5U_Epo_1-Stitching-09.0', '1U_Epo_1-Stitching-01.2']

    """
    filenames = os.listdir(LABEL_PATH)
    jpg_filenames = [filename for filename in filenames if
                     filename.endswith(".jpg")]  # point5U_Epo_3-Stitching-11.1.crops.clas.npy.gz.jpg
    basenames = [os.path.splitext(filename)[0] for filename in
                 jpg_filenames]  # point5U_Epo_3-Stitching-11.1.crops.clas.npy.gz
    SCENES = [x.replace(IMG_SUFFIX,"") for x in basenames]  #  point5U_Epo_3-Stitching-11.1
    SAMPLES = [get_sampleName_from_sceneName(x) for x in SCENES]  # point5U_Epo_3-Stitching-11
    return SAMPLES, SCENES


### START
SAMPLES, SCENES = get_samples_and_sceneIndexes_from_dir(LABEL_PATH)

rule targets:
    input:
        dag='dag.pdf',
        label_locs=expand('res/label_locs/{scene}.label.npy.gz',scene=SCENES),
        label_crops=expand('res/label_crops/{scene}.label.npy.gz',scene=SCENES),
        label_count="res/count.label.csv"


rule blob_detection:
    input:
        os.path.join(CZI_PATH, "{sample}.czi")
    output:
        npy = "res/blob_locs/{sample}.{sceneIndex}.locs.npy.gz",
        jpg = "res/blob_locs/view/{sample}.{sceneIndex}.jpg"
    log:
        "res/blob_locs/log/{sample}.{sceneIndex}.log"
    benchmark:
        "res/blob_locs/log/{sample}.{sceneIndex}.benchmark"
    threads:
        1
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 6000  # peaks at 4.5G on example 2024/11/27
    shell:
        """
        python workflow/scripts/blob_detection.singleScene.py \
        -input {input} -sceneIndex {wildcards.sceneIndex} -config config.yaml -odir res/blob_locs &> {log}
        """


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
        'res/label_locs/{sample}.{sceneIndex}.label.npy.gz.log'
    shell:
        """
        python workflow/scripts/jpg2npy.py {input.jpg} {input.czi} {input.blob_locs} \
        {wildcards.sceneIndex} {output.label_locs} &> {log}
        """


rule locs2crops:
    input:
        label_locs='res/label_locs/{sample}.{sceneIndex}.label.npy.gz',
        czi="data/czi/{sample}.czi",
        config='config.yaml'
    output:
        npy='res/label_crops/{sample}.{sceneIndex}.label.npy.gz',
        txt='res/label_crops/{sample}.{sceneIndex}.label.npy.gz.txt'
    log:
        'res/label_crops/log/{sample}.{sceneIndex}.label.npy.gz.log'
    shell:
        "python workflow/scripts/blob_cropping.py -czi {input.czi} -locs {input.label_locs} -i {wildcards.sceneIndex} \
        -config config.yaml -o {output.npy} > {output.txt} 2> {log}"


rule aggr_label_count:
    input:
        expand('res/label_crops/{scene}.label.npy.gz.txt', scene = SCENES)
    output:
        "res/count.label.csv"
    threads:
        1
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 1000
    priority:
        100
    log:
        "res/count.label.csv.log"
    shell:
        """
        python workflow/scripts/aggr_label_count.py {input} {output} &> {log}
        """


rule Create_DAG:
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 1000,
    threads:
        1
    output:
        "dag.pdf",
        "rulegraph.pdf"
    log:
        "dag.log"
    shell:
        """
        snakemake -s workflow/jpg2npy.Snakefile --dag targets 2> {log} | dot -Tpdf > {output[0]} 2>> {log}
        snakemake  -s workflow/jpg2npy.Snakefile --rulegraph targets 2> {log}| dot -Tpdf > {output[1]} 2>> {log}
        """

rule reset:
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 1000,
    threads:
        1
    shell:
        """
        rm -f dag.pdf rulegraph.pdf dag.log
        rm -f res/count.label.csv count.label.csv.log
        rm -rf res/label_locs res/label_crops
        rm -rf .snakemake
        """
