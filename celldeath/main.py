#!/usr/bin/python


import argparse
import textwrap
from slicer import slice_img
from train import trainer
from predict import predictor
from utils import create_folder, move_files
import os

def slice():
    create_folder(args.test_path)
    slice_img(args.indir_slicing, args.train_path, args.n_tiles, 
                args.test, args.test_path, args.perc_test)

def train():
    trainer(args.indir, args.model, args.valid_pct, args.l_lr, args.u_lr, args.aug, 
            args.epochs, args.bs, args.dropout, args.wd, args.imagenet, 
            args.predict, args.predict_path)

def predict():
    predictor(args.path_pred)


if __name__ == '__main__':
    #formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=15, width=100)
    path = os.path.expanduser('~user')+'celldeath/'
    parser = argparse.ArgumentParser(prog='celldeath', 
            formatter_class=argparse.RawTextHelpFormatter,
            description=textwrap.dedent('''
CellDeath is a simple deep learning script that trains light transmitted microscopy
images and then predict if those images contains cells undergoing cell death/apoptosis.

Subcommands are:

    slice: 
            Only needed to run once, and only if you need to slice your images. 

    train: 
            Core option for training the neural network. 
    
    predict: 
            Option for prediction. One or more images are given and yields prediction about if 
            those cells are undergoing cell death.  

We provide a pretrained model as an example. You can run your images with it, although prediction will be innacurate. For high accuracy, train your model with your own images. 
'''))
    subparser = parser.add_subparsers(title='commands', dest='command')
    
    parser_a = subparser.add_parser('slice')
    parser_a.add_argument('-indir_slicing', dest='indir_slicing', metavar='PATH',
                            help='Folder where images are stored.')
    parser_a.add_argument('-train_path', dest='train_path', metavar='PATH', 
                            default='img_split_train',
                            help='Folder where slice images are saved. Default is ~/celldeath/split_img')    
    parser_a.add_argument('-n_tiles', dest='n_tiles', metavar='INT',
                            default=4, type=int, choices=[2,4,6,8],
                            help='Number of tiles that will be generated. Default is 4; allowed values are 2,4,6 and 8.')
    parser_a.add_argument('-test', dest='test', action='store_true', 
                            help='Create a random separate set of images for testing. Default is False.')
    parser_a.add_argument('-test_path', dest='test_path', 
                            default='img_split_test', 
                            help='Path where images for testing will be stored. Default is img_split_test.')
    parser_a.add_argument('-perc_test', dest='perc_test', type=float, 
                            default=0.2, 
                            help='Percentage of iamges that will be used for testing. Default is 0.2.')
                
    
    parser_b = subparser.add_parser('train')
    parser_b.add_argument('-indir', metavar='PATH',
                            default='img_split_train',
                            help='Folder where images are stored. Beaware that default is with splitted images and so default is /split_img')
    parser_b.add_argument('-model', dest='model', action='store', default='resnet50',
                            choices=['resnet34', 'resnet50', 'resnet101', 'densenet121'],
                            help='Model used for training. Default is ResNet50..')
    parser_b.add_argument('-valid_pct', dest='valid_pct', type=float, 
                            default=0.2,
                            help='Validation percentage. Default is 0.2')
    parser_b.add_argument('-l_lr', dest='l_lr', type=float, metavar='FLOAT',
                            default=1e-4,
                            help='Lower Limit for learning rate.')
    parser_b.add_argument('-u_lr', dest='u_lr', type=float, metavar='INT',
                            default=1e-3,
                            help='Upper limit for learning rate.')
    parser_b.add_argument('-aug', dest='aug', action='store_true',
                            help='Add image augmentation. Default False')
    parser_b.add_argument('-epochs', dest='epochs', metavar='INT',
                            type=int, default=50, 
                            help='Number of epochs. Default is 50. ')
    parser_b.add_argument('-bs', dest='bs', type=int, metavar='INT',
                            default=16,   
                            help='Batch Size')
    parser_b.add_argument('-dropout', dest='dropout', type=float, default = 0.5,  
                            help='Drop out to be applied.')
    parser_b.add_argument('-wd', dest='wd', type=float, metavar='FLOAT',
                            default=0.1, 
                            help='Weight decay. Default is 0.01')                               
    parser_b.add_argument('-imagenet',dest='imagenet', 
                            action='store_true', 
                            help='Option for training using Imganet pretrained weights. Default is False.')
    parser_b.add_argument('-predict', dest='predict', action='store_true', 
                            help='Option for predict images immediately after training. Default is False.')
    parser_b.add_argument('-predict_path', dest='predict_path',
                            default='img_split_test', 
                            help='Path where images for prediction are located.')

    parser_c = subparser.add_parser('predict')
    parser_c.add_argument('-path_pred',  dest ='path_pred',  
                            metavar='PATH',
                            help='Path where image/s to predict are stored.')
    # parser_c.add_argument('-example', dest='example',  
    #                        action='store_true', 
    #                        default=path+'celldeath/img_split_predict',
    #                        help='Use provided example pretrained model. Only used for demonstration purpose as results wil be higly innacurate unless your cell images are higly similar to the ones originally used for training.')

    args = parser.parse_args()

    if args.command == 'slice':
        print('\n')
        print('Original images will be slice in {} tiles and stored in a separate folder.'.format(args.n_tiles))
        print('\n')
        slice()
        if args.test == True:
            move_files(args.train_path, args.test_path, args.perc_test)
        print('Done.')

    elif args.command == 'train':
        train()

    elif args.command == 'predict':
        predict() 
