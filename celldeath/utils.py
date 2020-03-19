#!/usr/bin/python


import os
import shutil
import numpy as np



def create_folder(folder):
    '''
    Create a folder.
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)

def move_files(fromdir, todir, perc_test):
    '''
    Move a defined percentage of files. 
    '''
    files = os.listdir(fromdir)
    for f in files:
        if np.random.rand(1) < perc_test:
            shutil.move(fromdir+'/'+ f, todir+'/'+f)        

