"""
Train
"""


import os

configfile: "config.train.yaml"
singularity: "workflow/ccount.sif"

WKDIR=os.getcwd()
DATA_TRAIN=config['DATA_TRAIN']
DATA_VAL=config['DATA_VAL']
SAMSPLING_RATES = config['sampling_rates']
CCOUNT_CONFIG=config['CCOUNT_CONFIG']
REPS=config['REPS']

rule targets:
    input:
        evaluations=expand(
        'res/3_evaluation_on_validationSet/{rate}.{rep}.txt',
        rate=SAMSPLING_RATES,
        rep=REPS),
        curve = 'res/plots/saturation_analysis.pdf'

rule subsample:
    input:
        crop=DATA_TRAIN
    output:
        small_crop='res/0_trainingData_subsets/{rate}.{rep}.npy.gz'
    log:
        'res/0_trainingData_subsets/{rate}.{rep}.npy.gz.log'
    benchmark:
         'res/0_trainingData_subsets/{rate}.{rep}.npy.gz.benchmark'
    threads:
        1
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 16000
    shell:
        """
        python workflow/scripts/crops_sampling.py \
        -crops {input.crop} \
        -ratio {wildcards.rate} \
        -output {output.small_crop} \
        &> {log}
        """

rule train:
    input:
        small_crop='res/0_trainingData_subsets/{rate}.{rep}.npy.gz',
        val_crop=DATA_VAL
    output:
        weight=protected('res/1_trained_weights/{rate}.{rep}.weights.h5')
    log:
        'res/1_trained_weights/{rate}.{rep}.weights.h5.log'
    benchmark:
         'res/1_trained_weights/{rate}.{rep}.weights.h5.benchmark'
    threads:
        16
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 4000
    shell:
        """
        python workflow/scripts/training.py \
        -crops_train {input.small_crop} \
        -crops_val {input.val_crop} \
        -config {CCOUNT_CONFIG} \
        -output {output.weight} \
        &> {log}
        """

rule classification:
    input:
        weight='res/1_trained_weights/{rate}.{rep}.weights.h5',
        data_val=DATA_VAL
    output:
        clas='res/2_count_on_validationSet/{rate}.{rep}.npy.gz' #todo:  test for non-complete scenes
    log:
        'res/2_count_on_validationSet/{rate}.{rep}.npy.gz.log'
    benchmark:
        'res/2_count_on_validationSet/{rate}.{rep}.npy.gz.benchmark'
    threads:
        2
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 8000
    shell:
        """
        python workflow/scripts/blob_classification.py \
        -crops {input.data_val} \
        -config {CCOUNT_CONFIG} \
        -weight {input.weight} \
        -output {output.clas} \
        &> {log}
        """

rule evaluation:
    input:
        clas='res/2_count_on_validationSet/{rate}.{rep}.npy.gz',
        data_val=DATA_VAL
    output:
        'res/3_evaluation_on_validationSet/{rate}.{rep}.txt'
    params:
        plot='res/3_evaluation_on_validationSet/{rate}.{rep}.pdf'
    log:
        'res/3_evaluation_on_validationSet/{rate}.{rep}.txt.log'
    benchmark:
        'res/3_evaluation_on_validationSet/{rate}.{rep}.txt.benchmark'
    threads:
        2
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 16000
    priority:
        100
    shell:
        """
        python workflow/scripts/eval_classification.py \
        -truth {input.data_val}  -pred {input.clas} \
        -output {params.plot} > {output} 2> {log}
        """

rule saturation_plot:
    input:
        expand('res/3_evaluation_on_validationSet/{rate}.{rep}.txt', rate=SAMSPLING_RATES, rep=REPS)
    output:
        'res/plots/saturation_analysis.pdf'
    log:
        'res/plots/saturation_analysis.pdf.log'
    threads:
        1
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 2000
    shell:
        """
        python workflow/scripts/plot_saturation.py \
        {input} {output}  &>{log}
        """

rule reset:
    shell:
        """
        echo 'deleting files..'
        rm -rf res/ lsf.log  log/ train.log report.html

        echo 'unlocking dir..'
        snakemake -s workflow/train.Snakefile -j 1 --unlock
        """
