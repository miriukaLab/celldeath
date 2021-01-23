#!/usr/bin/python



from fastai.vision import *
from fastai.callbacks import *
from fastai.metrics import error_rate
from PIL import Image, ImageFile
from pathlib import Path
import os
import time
import numpy as np
from celldeath.predict import predictor
from celldeath.utils import create_folder, extractMax, plot_confusion_matrix
import matplotlib.pyplot as plt
    

def trainer(indir, labels, model, valid_pct, l_lr, u_lr, aug, epochs, bs, dropout, wd, imagenet, test_path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    timestr = time.strftime("%Y%m%d-%H%M%S")  
    home_dir= str(Path.home())
    fnames = get_image_files(indir)
    pat = r'(?=('+'|'.join(labels)+r'))'
    tfms = get_transforms(do_flip=True, flip_vert=True, max_lighting=0.1, max_warp=0.)
    data = ImageDataBunch.from_name_re(indir, 
                                    fnames, 
                                    pat,
                                    ds_tfms=tfms, 
                                    valid_pct=valid_pct,
                                    bs=bs
                                    )                               
    create_folder(home_dir+'/celldeath/')
    if imagenet == False:
        stats=data.batch_stats()
        data.normalize(stats)
        if model == 'resnet50':
            learn = cnn_learner(data, models.resnet50, ps=dropout, metrics=accuracy)
        elif model == 'resnet34':
            learn = cnn_learner(data, models.resnet34, ps=dropout, metrics=accuracy)
        elif model == 'resnet101':
            learn = cnn_learner(data, models.resnet101, ps=dropout, metrics=accuracy)
        elif model == 'densenet121':
            learn = cnn_learner(data, models.densenet121, ps=dropout, metrics=accuracy)
        learn.fit_one_cycle(epochs, max_lr=slice(l_lr,u_lr), wd=wd,
                            callbacks=[ SaveModelCallback(learn, every='improvement', 
                                            monitor='valid_loss', name='best'),
                                        CSVLogger(learn, filename=home_dir+'/celldeath/history_'+model+'_'+timestr)
                                        ]
                            )
        learn.save('cell_death_training_'+timestr)
        learn.export()
        fig1 = learn.recorder.plot_losses()
        fig1 = plt.gcf()
        plt.savefig(home_dir+'/celldeath/'+'LossCurve_'+timestr+'.pdf', 
            dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
        learn.recorder.plot_metrics()
        plt.savefig(home_dir+'/celldeath/'+'Accuracy'+timestr+'.pdf', dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
        interp = ClassificationInterpretation.from_learner(learn)
        print('\n')
        
        for filename in os.listdir(home_dir+'/celldeath/'):
            if filename.startswith('history'):
                acc_valid,t_loss,v_loss=extractMax()
        
        interp.plot_confusion_matrix(return_fig=False)
        plt.tight_layout()
        plt.savefig(home_dir+'/celldeath/'+'Validation_confusion_matrix_'+timestr+'.pdf')

        print('Final Training Results for validation images:')
        print('\n')
        print('Accuracy:\t {:0.4f}'.format(acc_valid))
        print('Training Loss:\t {:0.4f}'.format(t_loss))
        print('Validation Loss:\t {:0.4f}'.format(v_loss))
        print('\n')
        
        time.sleep(2)
        acc_test = ''
        if test_path is not None:
            count_true = 0
            count_false = 0
            learn = load_learner(indir)
            word_to_id = {label: idx for idx, label in enumerate(set(labels))} 
            N = len(labels)
            matrix = np.zeros((N,N))
            for filename in os.listdir(test_path):
                img = open_image(test_path+'/'+filename) 
                pred_class,pred_idx,outputs = learn.predict(img)
                index = word_to_id[str(pred_class)]
                if str(pred_class) in filename:
                    prediction = 'True'
                    count_true += 1
                    matrix[index, index] += 1
                else:
                    prediction = 'False'
                    count_false += 1
                    for lab in labels:
                        if filename.startswith(lab):
                            outdex=word_to_id[lab]
                            matrix[outdex, index] += 1
                print('Image {}\tpredicts to\t{}\t{}'.format(filename, pred_class, prediction))
                plot_confusion_matrix(matrix,home_dir,labels,normalize=False)
            print('\n')
            acc_test = count_true/(count_true+count_false)  
            print('Accuracy for test images:\t {}\n'.format(acc_test))
        f = open(home_dir+'/celldeath/'+'report_'+timestr+'.txt', 'w+')
        f.write('Training parameters:\n\n')
        f.write(' indir: {}\n model: {}\n valid_pct: {}\n l_lr: {}\n u_lr: {}\n aug: {}\n epochs: {}\n bs: {}\n dropout: {}\n wd: {}\n imagenet: {}\n test_path: {}\n\n'.format(indir, model, valid_pct, l_lr, u_lr, aug, epochs, bs, dropout, wd, imagenet, test_path))
        f.write('\nFinal Training Results\n\n')
        f.write('Accuracy:\t\t {:0.4f}\n'.format(acc_valid))
        f.write('Training Loss:\t\t {:0.4f}\n'.format(t_loss))
        f.write('Validation Loss:\t\t {:0.4f}\n'.format(v_loss))
        f.write('\nAccuracy on the test set:\t {}\n'.format(acc_test))
        f.close()
        
    else:
        data.normalize(imagenet_stats)
        if model == 'resnet50':
            learn = cnn_learner(data, models.resnet50, ps=dropout, metrics=accuracy)
        elif model == 'resnet34':
            learn = cnn_learner(data, models.resnet34, ps=dropout, metrics=accuracy)
        elif model == 'resnet101':
            learn = cnn_learner(data, models.resnet101, ps=dropout, metrics=accuracy)
        elif model == 'densenet121':
            learn = cnn_learner(data, models.densenet121, ps=dropout, metrics=accuracy)
        learn.fit_one_cycle(1, 1e-2)
        learn.unfreeze()
        learn.fit_one_cycle(epochs, max_lr=slice(l_lr,u_lr), wd=wd,
                            callbacks=[SaveModelCallback(learn, every='improvement', 
                                            monitor='valid_loss', name='best'),
                                        CSVLogger(learn, filename=home_dir+'/celldeath/history_'+model+'_'+timestr)
                                        ]
                            )
        learn.save('cell_death_training_'+timestr)
        learn.export()
        fig1 = learn.recorder.plot_losses()
        fig1 = plt.gcf()
        plt.savefig(home_dir+'/celldeath/'+'LossCurve_'+timestr+'.pdf', 
            dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
        learn.recorder.plot_metrics()
        plt.savefig(home_dir+'/celldeath/'+'Accuracy'+timestr+'.pdf', dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
        interp = ClassificationInterpretation.from_learner(learn)
        print('\n')
        
        for filename in os.listdir(home_dir+'/celldeath/'):
            if filename.startswith('history'):
                acc_valid,t_loss,v_loss=extractMax()
        
        interp.plot_confusion_matrix(return_fig=False)
        plt.tight_layout()
        plt.savefig(home_dir+'/celldeath/'+'Validation_confusion_matrix_'+timestr+'.pdf')

        print('Final Training Results for validation images:')
        print('\n')
        print('Accuracy:\t {:0.4f}'.format(acc_valid))
        print('Training Loss:\t {:0.4f}'.format(t_loss))
        print('Validation Loss:\t {:0.4f}'.format(v_loss))
        print('\n')

        time.sleep(2)
        acc_test = ''
        if test_path is not None:
            count_true = 0
            count_false = 0
            learn = load_learner(indir)
            word_to_id = {label: idx for idx, label in enumerate(set(labels))} 
            N = len(labels)
            matrix = np.zeros((N,N))
            for filename in os.listdir(test_path):
                img = open_image(test_path+'/'+filename) 
                pred_class,pred_idx,outputs = learn.predict(img)
                index = word_to_id[str(pred_class)]
                if str(pred_class) in filename:
                    prediction = 'True'
                    count_true += 1
                    matrix[index, index] += 1
                else:
                    prediction = 'False'
                    count_false += 1
                    for lab in labels:
                        if filename.startswith(lab):
                            outdex=word_to_id[lab]
                            matrix[outdex, index] += 1
                print('Image {}\tpredicts to\t{}\t{}'.format(filename, pred_class, prediction))
                plot_confusion_matrix(matrix,home_dir,labels,normalize=False)
            print('\n')
            acc_test = count_true/(count_true+count_false)  
            print('Accuracy for test images:\t {}\n'.format(acc_test))
        f = open(home_dir+'/celldeath/'+'report_'+timestr+'.txt', 'w+')
        f.write('Training parameters:\n\n')
        f.write(' indir: {}\n model: {}\n valid_pct: {}\n l_lr: {}\n u_lr: {}\n aug: {}\n epochs: {}\n bs: {}\n dropout: {}\n wd: {}\n imagenet: {}\n test_path: {}\n\n'.format(indir, model, valid_pct, l_lr, u_lr, aug, epochs, bs, dropout, wd, imagenet, test_path))
        f.write('\nFinal Training Results\n\n')
        f.write('Accuracy:\t\t {:0.4f}\n'.format(acc_valid))
        f.write('Training Loss:\t\t {:0.4f}\n'.format(t_loss))
        f.write('Validation Loss:\t\t {:0.4f}\n'.format(v_loss))
        f.write('\nAccuracy on the test set:\t {}\n'.format(acc_test))
        f.close()

