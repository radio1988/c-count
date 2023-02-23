from aicsimageio import AICSImage

def read_czi(fname, Format="2019"):
    '''
    input: fname of czi file
    output: image_obj
    '''
    fname=str(fname)
    Format=str(Format)
    print('read_czi:', fname)
    if fname.endswith('czi'):
        if Format == '2019':
            image_obj = AICSImage(fname)
        else:
            raise Exception("Format not accepted")
    elif fname.endswith('czi.gz'):
        raise Exception("todo")
    else:
        raise Exception("input czi/czi.gz file type error\n")

    return image_obj

def parse_image_obj (image_obj, i = 0, Format='2019'):
    '''
    input: image_obj
    output: image 2d np.array
    '''
    Format=str(Format)
    i = int(i)
    if Format == '2019':
        image_obj.set_scene(i)
        image_array = image_obj.get_image_data()
        image =  image_array[0,0,0,:,:]
    else:
        raise Exception("Format not accepted")
    return image

#def parse_image_arrays (image_arrays, i = 0,  Format = '2019'):
#    '''
#    image_arrays: output from read_czi
#    i: index of [0,1,2,3], only this image will be parsed
#    Format: e.g. 2019
#    '''
#    import numpy as np
#    import gc
#
#    i = int(i)
#    Format = str(Format).strip()
#    if Format == "2018":
#        raise Exception("Format not accepted")
#        image = image_arrays[0, 1, 0, 0, :, :, 0]  # old CziFile package format
#        return image 
#    elif Format == "2019":        
#        # todo: Find Box faster by https://kite.com/python/docs/PIL.Image.Image.getbbox  
#        image = image_arrays[i, 0, :,  :, 0] # 0s
#        nz_image = np.nonzero(image)  # process_time(),36s, most time taken here, 1.4GB RAM with tracemalloc
#        nz0 = np.unique(nz_image[0]) # 1.5s
#        nz1 = np.unique(nz_image[1]) # 2.4s
#        del nz_image
#        n = gc.collect()
#        if len(nz0) < 2 or len(nz1) < 2: 
#            import warnings
#            warnings.warn('area'+str(i)+'is blank')
#            return False
#        image = image[min(nz0):max(nz0), min(nz1):max(nz1)]  # 0s
#        n_white = sum(sum(image>65534))
#        n_pixels = image.shape[0] * image.shape[1]
#        if n_white > 0:
#        #    image[image>65534] = 1 # remove very bright outlier pixels, blackImageBug
#            print(n_white, 'white pixels converted to black, which is {0:.0%} percent'.format(n_white/n_pixels))
#        return image
#        
#        # if concatenation:
#        #     # padding
#        #     heights = [x.shape[0] for x in lst]
#        #     widths = [x.shape[1] for x in lst]
#        #     max(heights)
#        #     max(widths)
#        #     for (i,image) in enumerate(lst):
#        #         print(image.shape, i)
#        #         pad_h = max(heights) - image.shape[0]
#        #         pad_w = max(widths) - image.shape[1]
#        #         lst[i] = np.pad(image, [[0,pad_h],[0,pad_w]], "constant")
#                
#        #     # concat: use a long wide image instead to adjust for unknown number of scanns
#        #     image = np.hstack(lst)
#        #     print("shape of whole picture {}: {}\n".format(fname, image.shape))
#        #     return image
#        # else:
#        #     # return a list of single are images
#        #     return lst #[image0, image1, image2 ..]
#    else:
#        raise Exception("image format error:", Format, "\n")
#        return None
