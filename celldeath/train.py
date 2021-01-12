#!/usr/bin/python



from fastai.vision import *
from fastai.callbacks import *
from fastai.metrics import error_rate
from PIL import Image, ImageFile
from pathlib import Path
import os
import time
from celldeath.predict import predictor
from celldeath.utils import create_folder
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
                                        CSVLogger(learn, filename='history'+timestr),
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
        interp = ClassificationInterpretation.from_learner(learn)
        print('\n')
        cm = interp.confusion_matrix()
        acc_valid = (cm[0,0] + cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
        pre = cm[0,0]/(cm[0,0]+cm[0,1])
        rec = cm[0,0]/(cm[0,0]+cm[1,0])
        print('Final Training Results for validation images:')
        print('\n')
        print('Accuracy:\t {:0.4f}'.format(acc_valid))
        print('Precision:\t {:0.4f}'.format(pre))
        print('Recall:\t\t {:0.4f}'.format(rec))
        print('\n')
        print('True Positives:\t\t {}'.format(cm[0,0]))
        print('False Positives:\t {}'.format(cm[0,1]))
        print('False Negatives:\t {}'.format(cm[1,0]))
        print('True Negatives:\t\t {}'.format(cm[1,1]))
        print('\n')
        time.sleep(2)
        acc_test = ''
        if test_path is not None:
            count_true = 0
            count_false = 0
            learn = load_learner(indir)
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
            acc_test = count_true/(count_true+count_false)  
            print('Accuracy for test images:\t {}\n'.format(acc_test))
        create_folder(home_dir+'/celldeath/')
        f = open(home_dir+'/celldeath/'+'report_'+timestr+'.txt', 'w+')
        f.write('Training parameters:\n\n')
        f.write(' indir: {}\n model: {}\n valid_pct: {}\n l_lr: {}\n u_lr: {}\n aug: {}\n epochs: {}\n bs: {}\n dropout: {}\n wd: {}\n imagenet: {}\n test_path: {}\n\n'.format(indir, model, valid_pct, l_lr, u_lr, aug, epochs, bs, dropout, wd, imagenet, test_path))
        f.write('\nFinal Training Results\n\n')
        f.write('Accuracy:\t\t {:0.4f}\n'.format(acc_valid))
        f.write('Precision:\t {:0.4f}\n'.format(pre))
        f.write('Recall:\t\t {:0.4f}\n'.format(rec))
        f.write('True Positives:\t\t {}\n'.format(cm[0,0]))
        f.write('False Positives:\t\t {}\n'.format(cm[0,1]))
        f.write('False Negatives:\t\t {}\n'.format(cm[1,0]))
        f.write('True Negatives:\t\t {}\n'.format(cm[1,1]))
        f.write('\nAccuracy on the test set:\t {}\n'.format(acc_test))
        f.close()
        interp.plot_confusion_matrix(return_fig=False)
        plt.tight_layout()
        plt.savefig(home_dir+'/celldeath/'+'confusion_matrix_'+timestr+'.pdf')
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
                                        CSVLogger(learn, filename='history'+timestr)
                                        ]
                            )
        timestr = time.strftime("%Y%m%d-%H%M%S")
        learn.save('cell_death_training_'+timestr)
        learn.export()
        fig1 = learn.recorder.plot_losses()
        fig1 = plt.gcf()
        plt.savefig(home_dir+'/celldeath/'+'LossCurve_'+timestr+'.pdf', 
            dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
        interp = ClassificationInterpretation.from_learner(learn)
        print('\n')
        cm = interp.confusion_matrix()
        acc_valid = (cm[0,0] + cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
        pre = cm[0,0]/(cm[0,0]+cm[0,1])
        rec = cm[0,0]/(cm[0,0]+cm[1,0])
        print('Final Training Results for validation images:')
        print('\n')
        print('Accuracy:\t {:0.4f}'.format(acc_valid))
        print('Precision:\t {:0.4f}'.format(pre))
        print('Recall:\t\t {:0.4f}'.format(rec))
        print('\n')
        print('True Positives:\t\t {}'.format(cm[0,0]))
        print('False Positives:\t {}'.format(cm[0,1]))
        print('False Negatives:\t {}'.format(cm[1,0]))
        print('True Negatives:\t\t {}'.format(cm[1,1]))
        print('\n')
        time.sleep(2)
        acc_test = ''
        if test_path is not None:
            count_true = 0
            count_false = 0
            learn = load_learner(indir)
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
            acc_test = count_true/(count_true+count_false)  
            print('Accuracy for test images:\t {}\n'.format(acc_test))
        create_folder(home_dir+'/celldeath/')
        f = open(home_dir+'/celldeath/'+'report_'+timestr+'.txt', 'w+')
        f.write('Training parameters:\n\n')
        f.write(' indir: {}\n model: {}\n valid_pct: {}\n l_lr: {}\n u_lr: {}\n aug: {}\n epochs: {}\n bs: {}\n dropout: {}\n wd: {}\n imagenet: {}\n test_path: {}\n\n'.format(indir, model, valid_pct, l_lr, u_lr, aug, epochs, bs, dropout, wd, imagenet, test_path))
        f.write('\nFinal Training Results\n\n')
        f.write('Accuracy:\t\t {:0.4f}\n'.format(acc_valid))
        f.write('Precision:\t {:0.4f}\n'.format(pre))
        f.write('Recall:\t\t {:0.4f}\n'.format(rec))
        f.write('True Positives:\t\t {}\n'.format(cm[0,0]))
        f.write('False Positives:\t\t {}\n'.format(cm[0,1]))
        f.write('False Negatives:\t\t {}\n'.format(cm[1,0]))
        f.write('True Negatives:\t\t {}\n'.format(cm[1,1]))
        f.write('\nAccuracy on the test set:\t {}\n'.format(acc_test))
        f.close()
        interp.plot_confusion_matrix(return_fig=False)
        plt.tight_layout()
        plt.savefig(home_dir+'/celldeath/'+'confusion_matrix_'+timestr+'.pdf')


