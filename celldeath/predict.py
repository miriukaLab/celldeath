#!/usr/bin/python

#to be done:
    #  add our trained model
    # add path to user trained model. 


from fastai.vision import *
import os
import re


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def predictor(path_pred):
    #if example == True:
    count_true = 0
    count_false = 0
    global accu
    accu = 0
    learn = load_learner('/home/smiriuka/celldeath/celldeath/img_split_train') # path to pretrained (by us) model 
    for filename in os.listdir(path_pred):
        img = open_image(path_pred+'/'+filename) 
        #if img.shape != ([3, 480, 640]):
            #   print('Image is not of the same size than those used for training!!')
            #  print('Trye to reconvert, though results may be suboptimal.')
            # img.resize([3, 480, 640])
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
    accu = acc_pred
    print('Accuracy:\t {}\n'.format(acc_pred))
    return acc_pred
    # else:
    #     count_true = 0
    #     count_false = 0
    #     learn = load_learner('/home/smiriuka/celldeath/celldeath/img_split_train') # path to user trained model 
    #     for filename in os.listdir(path_pred):
    #         whatis = []
    #         img = open_image(path_pred+'/'+filename) 
    #         #if img.shape != ([3, 480, 640]):
    #          #   print('Image is not of the same size than those used for training!!')
    #           #  print('Trye to reconvert, though results may be suboptimal.')
    #            # img.resize([3, 480, 640])
    #         pred_class,pred_idx,outputs = learn.predict(img)
    #         if str(pred_class) in filename:
    #             prediction = 'True'
    #             count_true += 1
    #         else:
    #             prediction = 'False'
    #             count_false += 1
    #         print('Image {}\tpredicts to\t{}\t{}'.format(filename, pred_class, prediction))
    #     print('\n')    
    #     print('Accuracy:\t {}\n'.format(count_true/(count_true+count_false)))
        
    
