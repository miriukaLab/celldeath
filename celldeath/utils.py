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

def plot_confusion_matrix(cm, dir,
                          target_names,
                          title='Test Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt  
    import itertools
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    if len(target_names) == 2:
        plt.ylim([1.5, -0.5])
    elif len(target_names) == 14:
        plt.ylim([13.5, -0.5])
    else:
        pass

    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(dir+'/celldeath/'+'Test_confusion_matrix_'+timestr+'.pdf')


def extractMax(csv):
    '''Extract values from csv file and store as variables'''
    import pandas as pd
    import numpy
    data = pd.read_csv(csv)
    data_max=data[data['accuracy']==data['accuracy'].max()]
    acc,tloss,vloss=(data_max['accuracy'],data_max['train_loss'],data_max['valid_loss'])
    acc=acc.values[0]
    tloss=tloss.values[0]
    vloss=vloss.values[0]
    return acc,tloss,vloss
