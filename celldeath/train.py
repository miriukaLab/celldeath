#!/usr/bin/python



from fastai.vision import *
from fastai.callbacks import *
from fastai.metrics import error_rate
from fastai.callbacks import *
#from fastai.callbacks.tensorboard import *
from PIL import ImageFile
import os
import time
from predict import predictor
from utils import create_folder
import matplotlib.pyplot as plt
    

def trainer(indir, model, valid_pct, l_lr, u_lr, aug, epochs, bs, dropout, wd, imagenet, test_path):
    #path_img = '/home/smiriuka/celldeath/celldeath/1hSliced'
    timestr = time.strftime("%Y%m%d-%H%M%S")  
    home_dir= os.path.expanduser('~user')
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    fnames = get_image_files(indir)
    pat = r'.*(CPT|DMSO).*'
    #pat = r'.*(control|celldeath).*'
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
                                        #TrackerCallback(learn, monitor=['valid_acc','valid_loss'])
                                        ]
                            )
        #learn.callback_fns.append(partial(LearnerTensorboardWriter(learn, name='run'+timestr)))
        #path = home_dir+'/apoptosis/celldeath/'+'cell_death_training_'+timestr
        learn.save('cell_death_training_'+timestr)
        learn.export()
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
            predictor(test_path)
            acc_test = os.environ['acc_pred']
        f = open('/home/smiriuka/celldeath/celldeath/reports/'+'report_'+timestr+'.txt', 'w+')
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
        plt.savefig('/home/smiriuka/celldeath/celldeath/reports/'+'confusion_matrix_'+timestr+'.pdf')
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
        #project_id = 'projct1'
        #tboard_path = Path('celldeath/tensorboard/' + project_id)
        #tensorboard --logdir=tboard_path --port=6006
        #os.system('python -m tensorflow.tensorboard --logdir=' + 'celldeath/tensorboard/')
        learn.fit_one_cycle(epochs, max_lr=slice(l_lr,u_lr), wd=wd,
                            callbacks=[SaveModelCallback(learn, every='improvement', 
                                            monitor='valid_loss', name='best'),
                                        CSVLogger(learn, filename='history'+timestr),
                                        #TrackerCallback(learn),
                                        #LearnerTensorboardWriter(learn, base_dir=tboard_path, name='run')
                                        ]
                            )
        timestr = time.strftime("%Y%m%d-%H%M%S")
        learn.save('cell_death_training_'+timestr)
        learn.export()
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
            predictor(test_path)
            acc_test = os.environ['acc_pred']
        f = open('/home/smiriuka/celldeath/celldeath/reports/'+'report_'+timestr+'.txt', 'w+')
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
        plt.savefig('/home/smiriuka/celldeath/celldeath/reports/'+'confusion_matrix_'+timestr+'.pdf')


