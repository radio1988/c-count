rule blob_detection:
    input:
        os.path.join(config['DATA_DIR'], "{s}.czi")
    output:
        touch("res/blob_locs/{s}.done"),
        #"res/blob_locs/{s}.{i}.crops.npy.gz"  # uncertain num of scenes (4 or less)
    threads:
        1
    resources:
        mem_mb=lambda wildcards, attempt: attempt * 8000  # 4.5G at 2024/11 on example
    log:
        "log/blob_locs/{s}.log"
    benchmark:
        "log/blob_locs/{s}.benchmark"
    shell:
        """
        # todo: dynamic config.fname
        python workflow/scripts/blob_detection.py \
        -i {input} -c config.yaml -odir res/blob_locs &> {log}  
        """
