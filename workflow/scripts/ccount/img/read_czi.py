from aicsimageio import AICSImage

def read_czi(fname, Format="2019"):
    '''
    input: fname of czi file
    output: image_obj
    '''
    fname=str(fname)
    Format=str(Format)
    print('read_czi:', fname)
    print('Format', Format)
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
