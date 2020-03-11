#!/usr/bin/python


import os




def create_folder(folder):
    #home_dir= os.path.expanduser('~user') 
    if not os.path.exists(folder):
        os.makedirs(folder)


def crop_img(args):
    pass

def check_size(args):
    for filename in os.listdir(path):
        print(filename.shape)
