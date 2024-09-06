"""
Train
"""


import os

configfile: "config.train.yaml"

WKDIR=os.getcwd()
DATA_TRAIN=config['DATA_TRAIN']
DATA_VAL=config['DATA_VAL']
SAMSPLING_RATES = config['sampling_rates']
CCOUNT_CONFIG=config['CCOUNT_CONFIG']

rule targets:
    input:
        subsamples=expand('res/small_data/{rate}.npy.gz', rate=SAMSPLING_RATES), 
        weights=expand('res/weights/{rate}.weights.h5', rate=SAMSPLING_RATES), 
        classifications=expand('res/clas/{rate}.npy.gz', rate=SAMSPLING_RATES),
        evaluations=expand('res/eval/{rate}.txt', rate=SAMSPLING_RATES),
        # curve = 'data_curve.pdf'

rule subsampling: 
    input:
        crop=DATA_TRAIN
    output:
        small_crop='res/small_data/{rate}.npy.gz'
    log:
        'res/small_data/{rate}.npy.gz.log'
    benchmark:
         'res/small_data/{rate}.npy.gz.benchmark'
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

rule training: 
    input:
        small_crop='res/small_data/{rate}.npy.gz',
        val_crop=DATA_VAL
    output:
        weight='res/weights/{rate}.weights.h5'
    log:
        'res/weights/{rate}.weights.h5.log'
    benchmark:
         'res/weights/{rate}.weights.h5.benchmark'
    threads:
        4
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 15000
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
        weight='res/weights/{rate}.weights.h5',
        data_val=DATA_VAL
    output:
        clas='res/clas/{rate}.npy.gz' # test
    log:
        'res/clas/{rate}.npy.gz.log'
    benchmark:
        'res/clas/{rate}.npy.gz.benchmark'
    threads:
        2
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 8000
    shell:
        """
        python workflow/scripts/classification.py \
        -crops {input.data_val} \
        -config {CCOUNT_CONFIG} \
        -weight {input.weight} \
        -output {output.clas} \
        &> {log}
        """

rule evaluation:
    input:
        clas='res/clas/{rate}.npy.gz',
        data_val=DATA_VAL
    output:
        eval='res/eval/{rate}.txt'
    log:
        'res/eval/{rate}.txt.log'
    benchmark:
        'res/eval/{rate}.txt.benchmark'
    threads:
        1
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 8000
    priority:
        100
    shell:
        """
        python workflow/scripts/eval_classification.py \
        {input.clas} \
        {input.data_val} \
        1>{output} \
        2>{log}
        """
