#!/usr/bin/python


from fastai.vision import *
import os
import re


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def predictor(path_pred):
    '''
    Predict on given images. 
    Put your images in a folder calles 'predict', and this should be a subfolder 
    of the indir folder used to train.  
    '''
    count_true = 0
    count_false = 0
    learn = load_learner(path_pred)
    for filename in os.listdir(path_pred+'/predict'):
        img = open_image(path_pred+'/predict'+'/'+filename) 
        pred_class,pred_idx,outputs = learn.predict(img)
        if str(pred_class) in filename:
            prediction = 'True'
            count_true += 1
        else:
            prediction = 'False'
            count_false += 1
        print('Image {}\tpredicts to\t{}\t{}'.format(filename, pred_class, prediction))
    print('\n')
    acc_pred = count_true/(count_true+count_false)  
    print('Accuracy for test images:\t {}\n'.format(acc_pred))
    os.environ['acc_pred'] = str(acc_pred)
    return acc_pred
    
