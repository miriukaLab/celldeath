#!/usr/bin/python



from fastai.vision import *
from fastai.callbacks import *
from fastai.metrics import error_rate
from PIL import ImageFile
import os
import time
from utils import create_folder
    


def trainer(indir, model, valid_pct, l_lr, u_lr, aug, epochs, bs, dropout, wd, imagenet):
    #path_img = '/home/smiriuka/celldeath/celldeath/1hSliced'
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
        learn.fit_one_cycle(epochs, max_lr=slice(l_lr,u_lr), wd=wd)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        #home_dir= os.path.expanduser('~user')
        #path = home_dir+'/apoptosis/celldeath/'+'cell_death_training_'+timestr
        learn.save('cell_death_training_'+timestr)
        learn.export()
        interp = ClassificationInterpretation.from_learner(learn)
        print('\n')
        cm = interp.confusion_matrix()
        accu = (cm[0,0] + cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
        pre = cm[0,0]/(cm[0,0]+cm[0,1])
        rec = cm[0,0]/(cm[0,0]+cm[1,0])
        print('Final Training Results:')
        print('\n')
        print('Accuracy:\t {:0.4f}'.format(accu))
        print('Precision:\t {:0.4f}'.format(pre))
        print('Recall:\t\t {:0.4f}'.format(rec))
        print('\n')
        print('True Positives:\t\t {}'.format(cm[0,0]))
        print('False Positives:\t {}'.format(cm[0,1]))
        print('False Negatives:\t {}'.format(cm[1,0]))
        print('True Negatives:\t\t {}'.format(cm[1,1]))
        print('\n')
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
        learn.fit_one_cycle(epochs, max_lr=slice(l_lr,u_lr), wd=wd)
        learn.fit_one_cycle(epochs, max_lr=slice(l_lr,u_lr), wd=wd)
        #learn.fit_one_cycle(epochs, max_lr=slice(1e-4,1e-3), wd=0.1)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        learn.save('cell_death_training_'+timestr)
        learn.export()
        interp = ClassificationInterpretation.from_learner(learn)
        print('\n')
        cm = interp.confusion_matrix()
        accu = (cm[0,0] + cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
        pre = cm[0,0]/(cm[0,0]+cm[0,1])
        rec = cm[0,0]/(cm[0,0]+cm[1,0])
        print('Final Training Results:')
        print('\n')
        print('Accuracy:\t {:0.4f}'.format(accu))
        print('Precision:\t {:0.4f}'.format(pre))
        print('Recall:\t\t {:0.4f}'.format(rec))
        print('\n')
        print('True Positives:\t\t {}'.format(cm[0,0]))
        print('False Positives:\t {}'.format(cm[0,1]))
        print('False Negatives:\t {}'.format(cm[1,0]))
        print('True Negatives:\t\t {}'.format(cm[1,1]))
        print('\n')
    

    #interp = ClassificationInterpretation.from_learner(learn)
    #learn.fit_one_cycle(8)
    #learn.unfreeze()
    #learn.fit_one_cycle(40, max_lr=slice(1e-4,1e-3))


