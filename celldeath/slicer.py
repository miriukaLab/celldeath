
#!/usr/bin/python


import image_slicer
import os



# def split_image_in_four(indir, outdir):
#     for root, dirs, filenames in os.walk(indir):
#         for file in filenames:
#             if file.endswith('.png'):
#                 tiles = image_slicer.slice(os.path.join(indir,file), 4, save=False)
#                 image_slicer.save_tiles(tiles, directory=outdir, prefix=file)

def slice_img(indir_slicing, outdir_slicing, n_tiles):
    '''
    Split images in n tiles. Default is 4. 
    Supports .png and .jpg files
    '''
    path = os.path.expanduser('~user')
    for root, dirs, filenames in os.walk(indir_slicing):
        for file in filenames:
            if file.endswith('.png'):
                tiles = image_slicer.slice(os.path.join(indir_slicing,file), n_tiles, save=False)
                image_slicer.save_tiles(tiles, directory=outdir_slicing, prefix=file)
            elif file.endswith('.jpg'):
                tiles = image_slicer.slice(os.path.join(indir_slicing,file), n_tiles, save=False)
                image_slicer.save_tiles(tiles, directory=outdir_slicing, prefix=file) 



