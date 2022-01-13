import sys
import pandas as pd

def get_areaDF_from_area_txts(path="classification1/area/"):
    '''
    input: path storing xxx.area.txt files
    output: data frame of #pixels for each scanned area (or any npy.gz output from blob detection)
    '''
    import os
    area_fnames = os.listdir(path)
    area_fnames = list(filter(lambda x: x.endswith("txt"), area_fnames))
    print('There are: ', len(area_fnames), 'areas scanned')

    labels0 = area_fnames.copy()
    labels0 = [x.replace(".area.txt", "") for x in labels0]
    # ['E2f4_CFUe_KO_1-Stitching-01.0',
    #  'E2f4_CFUe_KO_1-Stitching-01.1',
    #  'E2f4_CFUe_KO_1-Stitching-01.2']

    area_fnames = [path + x for x in area_fnames]
    # ['classification1/area/E2f4_CFUe_KO_1-Stitching-01.0.area.txt',
    #  'classification1/area/E2f4_CFUe_KO_1-Stitching-01.1.area.txt',
    #  'classification1/area/E2f4_CFUe_KO_1-Stitching-01.2.area.txt']
    
    areaLSF = []
    labels = []
    for i,f in enumerate(area_fnames):
        try:
            d = pd.read_table(f, header=None)
        except pd.errors.EmptyDataError:
            print(i, f,labels0[i], 'contains no blobs, bad');
            continue
        areaLSF.append( d.iloc[:, 0].tolist())
        print(i, f,labels0[i], 'good');
        labels.append(labels0[i])

    areaDF = pd.DataFrame(areaLSF).T
    areaDF = pd.DataFrame(areaDF)
    areaDF.columns = labels
    print("there are", len(labels), "areas aggregated")
    
    # 	E2f4_CFUe_KO_1-Stitching-01.0	E2f4_CFUe_KO_1-Stitching-01.1	E2f4_CFUe_KO_1-Stitching-01.2
    # 0	10747.0	4693.0	8281.0
    # 1	4801.0	4256.0	4431.0
    # 2	5023.0	3986.0	2939.0
    # 3	4790.0	3026.0	3452.0
    # 4	4173.0	4045.0	2997.0
    # ...	...	...	...
    # 413	NaN	NaN	NaN
    # 414	NaN	NaN	NaN
    # 415	NaN	NaN	NaN
    # 416	NaN	NaN	NaN
    # 417	NaN	NaN	NaN
    
    return(areaDF)


print("usage: get_areas.py path_to_area_files output_csv_fname")
print('example: python workflow/get_areas.py res/classification1/area/ res/areas.csv')
if len(sys.argv) < 3:
    sys.exit("cmd error")
areaDF = get_areaDF_from_area_txts(sys.argv[1])
print(areaDF.iloc[0:3, 0:2])
areaDF.to_csv(sys.argv[2], float_format='%i')
