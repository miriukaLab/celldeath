#!/usr/bin/python




from fastai.vision import *
import os
import re


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def predictor(path_pred, pretrained):
    if pretrained == True:
        count = 0
        count_zero = 0
        learn = load_learner('/home/smiriuka/apoptosis/celldeath/1hSliced/')
        for filename in os.listdir(path_pred):
            img = open_image(path_pred+filename)
            print(img.shape) 
            if img.shape != ([3,480,640]):
                print('Image is not of the same size than those used for training!!')
                print('Trye to reconvert, though results may be suboptimal.')
                img.resize([3,480,640])
            pred_class,pred_idx,outputs = learn.predict(img)
            pat1 = r'.*(DMSO).*'
            pat2 = r'.*(CPT).*'
            whatis = re.findall(str(pred_class), filename)
            if whatis[0] == str(pred_class):
                prediction = 'True'
                count += 1
            elif whatis[0] == str(pred_class):
                prediction = 'True'
                count += 1
            else:
                prediction = 'False'
                count_zero += 1
        print('Image {}\tpredicts to\t{}\t{}'.format(filename, pred_class, prediction))
        print('\n')    
        print('Accuracy:\t {}\n'.format(count/count+count_zero))
    else:
        count = 0
        count_zero = 0
        for filename in os.listdir(path_pred):
            img = open_image(path_pred+filename) 
            print(img.shape)
            pred_class,pred_idx,outputs = learn.predict(img)
            pat1 = r'.*(DMSO).*'
            pat2 = r'.*(CPT).*'
            whatis = re.findall(str(pred_class), filename)
            if whatis[0] == str(pred_class):
                prediction = 'True'
                count += 1
            elif whatis[0] == str(pred_class):
                prediction = 'True'
                count += 1
            else:
                prediction = 'False'
                count_zero += 1
            print('Image {}\tpredicts to\t{}\t{}'.format(filename, pred_class, prediction))
        print('\n')    
        print('Accuracy:\t {}\n'.format(count/count+count_zero))
        
    
