"""
COUNT

Input:
- czi: microscopic image for the plate including all scenes
- weight: trained ccount weight
- config: config.yaml used when weight was trained for neural network parameters

Steps:
- detect blobs in czi files
- classify blobs into positives and negatives
- count positives in each plate-scene
- aggregate all counts into a csv file

Output:
- res/COUNT.csv
-

"""

import os
from scripts.ccount_utils.snake import input_names
from scripts.ccount_utils.snake import get_samples
singularity: "workflow/ccount.sif"


configfile: "config.yaml"
DATA_DIR = config["DATA_DIR"]
FORMAT = config["FORMAT"]
WEIGHT = config["WEIGHT"]
WKDIR = os.getcwd()

SAMPLES = get_samples(DATA_DIR)
if len(SAMPLES) == 0:
    raise ValueError("No czi files found in DATA_DIR")

rule targets:
    input:
        # czi2img=expand("log/img/{s}.done", s=SAMPLES), # skipped, clas vis more helpful
        # blob_cropping=input_names(prefix="res/blob_crops/", SAMPLES=SAMPLES,
        #                          suffix='.crops.npy.gz'),
        # classification=input_names(prefix='res/count/', SAMPLES=SAMPLES,
        #                            suffix=".crops.clas.txt"),
        # filter_crops=input_names(prefix='res/count/pos/', SAMPLES=SAMPLES,
        #                          suffix=".crops.clas.npy.gz"),
        count_file='res/COUNT.csv',
        area1_agg="res/areas.csv",
        view_clas_on_image=input_names(prefix="res/count/",SAMPLES=SAMPLES,
            suffix=".crops.clas.npy.gz.jpg"),
        blob_locs=input_names(prefix="res/count/", SAMPLES=SAMPLES, suffix='.locs.clas.npy.gz'),
#        blob_locs=expand("res/count/{s}.{i}.locs.clas.npy.gz",s=SAMPLES,i=[0, 1, 2, 3]),
        rulegraph="rulegraph.pdf"
        # plot = "res/plots/areas.histogram.pdf"

include: 'rules/czi2img.smk'
include: 'rules/blob_detection.smk'


rule blob_cropping:
    input:
        czi=os.path.join(config['DATA_DIR'], "{s}.czi"),
        blob_locs_flag="res/blob_locs/{s}.done",
        #blob_locs="res/blob_locs/{s}.{i}.crops.npy.gz"
    output:
        temp('res/blob_crops/{s}.{i}.crops.npy.gz')
    threads:
        1
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 16000
    log:
        'log/blob_crops/{s}.{i}.crops.npy.gz.log'
    benchmark:
        'log/blob_crops/{s}.{i}.crops.npy.gz.benchmark'
    shell:
        """
        python workflow/scripts/blob_cropping.py -czi {input.czi} \
        -locs res/blob_locs/{wildcards.s}.{wildcards.i}.locs.npy.gz \
        -i {wildcards.i}  -config config.yaml -o {output} &> {log}
        """

rule classification:
    input:
        blob_crops='res/blob_crops/{s}.{i}.crops.npy.gz',
        weight=WEIGHT
    output:
        crops=temp("res/count/{s}.{i}.crops.clas.npy.gz"),
        locs="res/count/{s}.{i}.locs.clas.npy.gz",
        txt="res/count/{s}.{i}.crops.clas.txt"
    threads:
        1
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 8000
    log:
        "log/count/{s}.{i}.log"  # todo: log deleted if job fail
    benchmark:
        "log/count/{s}.{i}.benchmark"
    shell:
        """
        python workflow/scripts/classify.py  \
        -crops {input.blob_crops} -weight {input.weight} \
        -config config.yaml -output {output.crops} &> {log}
        """


rule aggr_count:
    input:
        input_names(SAMPLES=SAMPLES,prefix="res/count/",suffix=".crops.clas.txt")
    output:
        "res/COUNT.csv"
    threads:
        1
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 1000
    priority:
        100
    log:
        "log/COUNT.csv.log"
    shell:
        """
        python workflow/scripts/aggr_count.py {input} {output} &> {log}
        """

rule filter_crops:
    input:
        "res/count/{s}.{i}.crops.clas.npy.gz"
    output:
        "res/count/pos/{s}.{i}.crops.clas.npy.gz"
    threads:
        1
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 8000
    log:
        "log/count/pos/{s}.{i}.crops.clas.npy.gz.log"
    benchmark:
        "log/count/pos/{s}.{i}.crops.clas.npy.gz.benchmark"
    shell:
        """
        python workflow/scripts/crops_filtering.py -crops {input} \
        -label 1 -output {output} &> {log}
        """

rule view_clas_on_image:
    input:
        crop="res/count/{s}.{i}.crops.clas.npy.gz",
        czi=os.path.join(config['DATA_DIR'], "{s}.czi")
    output:
        "res/count/{s}.{i}.crops.clas.npy.gz.jpg"
    threads:
        1
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 16000
    log:
        "log/count/{s}.{i}.crops.clas.npy.gz.jpg.log"
    benchmark:
        "log/count/{s}.{i}.crops.clas.npy.gz.jpg.benchmark"
    shell:
        """
        python workflow/scripts/visualize_locs_on_czi.py \
        -crops {input.crop} \
        -index {wildcards.i} \
        -czi {input.czi} \
        -config config.yaml \
        -output {output} &> {log}
        """

rule area_calculation:
    input:
        'res/count/pos/{s}.{i}.crops.clas.npy.gz'
    output:
        txt='res/count/pos/area/{s}.{i}.area.txt',
        npy='res/count/pos/area/{s}.{i}.area.npy.gz'
    log:
        "res/count/pos/area/{s}.{i}.area.log"
    benchmark:
        "res/count/pos/area/{s}.{i}.area.benchmark"
    threads:
        1
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 8000
    shell:
        """
        python workflow/scripts/area_calculation.py -crops {input} -output {output.txt} &> {log}
        """

rule area_aggregation:
    '''
    Will aggreated all files under res/count/area, regardless of config.yaml
    '''
    input:
        input_names(SAMPLES=SAMPLES,
            prefix="res/count/pos/area/",suffix=".area.txt")
    output:
        "res/areas.csv"
    log:
        "res/areas.csv.log"
    threads:
        1
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 1000
    shell:
        """
        python workflow/scripts/aggr_area_info.py res/count/pos/area/ res/areas.csv &> {log}
        """


rule area_histogram:
    '''
    '''
    input:
        "res/areas.csv"
    output:
        "res/plots/areas.histogram.pdf"
    log:
        "res/plots/areas.histogram.pdf.log"
    threads:
        1
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 1000
    shell:
        """
        python workflow/scripts/plot_area_histogram.py {input} {output} &> {log}
        """

# rule view0:
#     input:
#         "res/blobs/{s}.done"
#     output:
#         html="res/blobs/view/{s}.{i}.html"
#     params:
#         html="../res/blobs/view/{s}.{i}.html"
#     log:
#         "log/blobs/view/{s}.{i}.html.log"
#     benchmark:
#         "log/blobs/view/{s}.{i}.html.benchmark"
#     threads:
#         1
#     resources:
#         mem_mb=lambda wildcards, attempt: attempt * 8000
#     shell:
#         """
#         fname="res/blobs/{wildcards.s}.{wildcards.i}.npy.gz" dir={WKDIR} \
#         jupyter nbconvert --to html --execute workflow/notebooks/viewing_blobs.ipynb \
#         --output {params.html} &> {log}
#         """
#
# rule view1:
#     input:
#         "res/count/{s}.{i}.clas.npy.gz"  # some will not exist, but just ignore warnings
#     output:
#         html="res/count/view/{s}.{i}.html"
#     params:
#         html="../res/count/view/{s}.{i}.html"
#     log:
#         "log/count/view/{s}.{i}.html.log"
#     benchmark:
#         "log/count/view/{s}.{i}.html.benchmark"
#     threads:
#         1
#     resources:
#         mem_mb=lambda wildcards, attempt: attempt * 8000
#     shell:
#         """
#         mkdir -p res res/count res/count/view
#         fname={input} dir={WKDIR} \
#         jupyter nbconvert --to html --execute workflow/notebooks/viewing_blobs.ipynb \
#         --output {params.html} &> {log}
#         """

rule Create_DAG:
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 1000,
    threads:
        1
    output:
        dag="workflow.pdf",
        rulegraph="rulegraph.pdf"
    log:
        "log/rulegraph.log"
    shell:
        "snakemake -s workflow/count.Snakefile --dag targets |dot -Tpdf > {output.dag} 2> {log};"
        "snakemake -s workflow/count.Snakefile  --rulegraph targets | dot -Tpdf > {output.rulegraph} 2>> {log}"



rule reset:
    shell:
        """
        echo 'deleting files..'
        rm -rf res/ lsf.log  log/ train.log workflow.pdf rulegraph.pdf

        echo 'unlocking dir..'
        snakemake -s workflow/count.Snakefile -j 1 --unlock
        """
