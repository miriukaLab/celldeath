#!/usr/bin/python


import image_slicer
import os



# def split_image_in_four(indir, outdir):
#     for root, dirs, filenames in os.walk(indir):
#         for file in filenames:
#             if file.endswith('.png'):
#                 tiles = image_slicer.slice(os.path.join(indir,file), 4, save=False)
#                 image_slicer.save_tiles(tiles, directory=outdir, prefix=file)

def slice_img(args):
    '''
    Split images in n tiles. Default is 4. 
    Supports .png and .jpg files
    '''
    path = os.path.expanduser('~user')
    for root, dirs, filenames in os.walk(args.indir):
        for file in filenames:
            if file.endswith('.png'):
                tiles = image_slicer.slice(os.path.join(args.indir,file), args.num_tiles, save=False)
                image_slicer.save_tiles(tiles, directory=args.outdir, prefix=file)
            elif file.endswith('.jpg'):
                tiles = image_slicer.slice(os.path.join(args.indir,file), args.num_tiles, save=False)
                image_slicer.save_tiles(tiles, directory=args.outdir, prefix=file) 


def crop_img(args):
    pass

def check_size(args):
    for filename in os.listdir(path):
        print(filename.shape)


