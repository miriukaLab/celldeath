
#!/usr/bin/python


import image_slicer
import os


def slice_img(indir_slicing, train_path, n_tiles, test, test_path, perc_test):
    '''
    Split images in n tiles. Default is 4. 
    Supports .png and .jpg files
    '''
    for root, dirs, filenames in os.walk(indir_slicing):
        for file in filenames:
            if file.endswith('.png'):
                tiles = image_slicer.slice(os.path.join(indir_slicing,file), n_tiles, save=False)
                image_slicer.save_tiles(tiles, directory=train_path, prefix=file)
            elif file.endswith('.jpg'):
                tiles = image_slicer.slice(os.path.join(indir_slicing,file), n_tiles, save=False)
                image_slicer.save_tiles(tiles, directory=train_path, prefix=file) 



