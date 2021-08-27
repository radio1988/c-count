def input_names(SAMPLES, words = ["Top", "Left", "Right", "Bottom"], 
    prefix = "res/blobs/view/", suffix = '.html', NUMS=[0,1,2,3]):
    """
    If any word in words found in sample_name (s), 
        only one output [0]
    else 
        four output [0,1,2,3]
    
    output example:
    ['res/blobs/view/E2f4_CFUe_WT3_3-Stitching-20.0.html', 'res/blobs/view/E2f4_CFUe_WT3_3-Stitching-20.1.html', 
    'res/blobs/view/E2f4_CFUe_WT3_3-Stitching-20.2.html', 'res/blobs/view/E2f4_CFUe_WT3_3-Stitching-20.3.html', 
    'res/blobs/view/E2f4_CFUe_WT3_3_Top-Stitching-21.0.html']
    """
    lst = []
    for s in SAMPLES:
        if any([w in s for w in words]):
            res = prefix + s + '.0' + suffix
            lst.append(res)
        else:
            res = map(lambda i: prefix + s + '.' + str(i) + suffix, NUMS)  # all four nums
            lst=lst+list(res)
    return lst