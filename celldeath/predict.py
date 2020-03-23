#!/usr/bin/python


from fastai.vision import *
import os
import re


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def predictor(test_path):
    '''
    Predict on given images. 
    '''
    count_true = 0
    count_false = 0
    learn = load_learner('/DATA/sgm/apoptosis/1hr_train')
    for filename in os.listdir(test_path):
        img = open_image(test_path+'/'+filename) 
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
    
