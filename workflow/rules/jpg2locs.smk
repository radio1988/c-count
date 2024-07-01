

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
