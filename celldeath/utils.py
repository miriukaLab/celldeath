#!/usr/bin/python


import os




def create_folder(folder):
    #home_dir= os.path.expanduser('~user') 
    if not os.path.exists(folder):
        os.makedirs(folder)

