def read_czi(fname, Format="2019"):
    '''
    input: fname of czi file
    output: 2d numpy array, uint8 for 2019
    assuming input czi format (n, 1, :, :, 1)
    e.g. (4, 1, 70759, 65864, 1)

    '''
    from czifile import CziFile
    
    fname=str(fname)
    Format=str(Format)
    print('read_czi:', fname)
    if fname.endswith('czi'):
        with CziFile(fname) as czi:
            image_arrays = czi.asarray()  # 129s, Current memory usage is 735.235163MB; Peak was 40143.710599MB
            print(image_arrays.shape)
    elif fname.endswith('czi.gz'):
        raise Exception("todo")
    else:
        raise Exception("input czi/czi.gz file type error\n")

    return image_arrays