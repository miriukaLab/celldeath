# **celldeath**

A simple python script based on deep learning optimized for classifying cell death in simple transmitted light microscopy images. 
<img src="./img.png" align="right" height="200" width="200" >

Our goal is to provide a tool for someone with no or minimal idea of deep learning.

## Getting Started

celldeath is a deep learning tool that trains with light transmitted microscopy images in order to predict cell death/apoptosis.

We aimed to keep it simple, so anyone can apply it with minimal deep learning knowledge. Parameters for training are already optimized for high accuracy. However, we provide some suggestions for fine tunning your training, as each experimental model may have unique values.

### Prerequisites

Please note that for training a deep learning model you need a GPU and a lot of images. If you don't have a GPU available, you can try in any cloud computing provider (AWS, Google Cloud, Azure, etc).

### Installing

For installation, we suggest create a virtual environment, and then:

```bash
pip install celldeath
```

### Dependencies

* fastai>=1.0.60
* image-slicer==0.3.0
* matplotlib>=3.1.1

## Usage

celldeath has three subcommands (*train*, *predict* and *slice*), each one with their options (see below). However, for a simpler use, you can follow these next steps. Defaults are optimized for early cell death recognition in transmitted light microscopy images. 

### simple use

Open your terminal and point to the celldeath folder. Then type providing type

```bash
celldeath slice -indir_slicing your/img/folder
```

Be aware that if you train several times with same images, this previous step has to be done just one time. This command will slice your images into 4 tiles,which then will be used for classification using the next command

```bash
celldeath train -labels your_labels -imagenet
```

You have to provide your labels, which must be included as a part of the filename of each image. Default values are *control* and *celldeath*. You can have more than 2 conditions. 

Default values should lead to high accuracy in your images for prediction. (Of course, your images should be good enough...) Once training is finished, you will find a .txt file under the folder *reports* that includes all training parameters, results on the validation image set, and accuracy on the test set. You will also get a confusion matrix with absolute numbers for prediction.  


## Subcommands

The next subcommands will give you some more control on preprocessing, training and prediction. Subcommands are:

* **celldeath slice** | Only needed to run once, and only if you wish to slice your images.

* **celldeath train** | Core subcommand for training the neural network.

* **celldeath predict** | One or more images are given and it returns prediction based on your previous training.  

### train

celldeath allows you to train your own images without too much knowledge of deep learning. Be aware that you need at least a few hundreds of images for proper training and prediction. However, with the subcommand *slice* you can split your images in n tiles (usually 4), and hence increase your training performance. Also, using the argument -aug will artificially increase your images by flipping and rotating them. The subcommand *train* already has most of the option set up for an excellent training, and so you can provide a minimal input (just the path where your images are stored) to get a high accuracy, providing that your images and your experiement are reasonable.

A few recommendations:  

* be consistent in your experimental cell conditions. 
* get as many images as possible all over the well.
* try both pretrained (-imagenet) and not pretrained. Pretrainng may be faster, but it is not necessary better though (yes, no cells in imagenet!).  
* try augmentation. We set up a few augmentation parameters (flip and rotation, minimal changes in light condition) since some other common arguments results in a lower performance.  
* batch size (-bs) will depend on you GPU and the size of your images.  
* Always try weight decay. We set it up to 0.01, but also try 0.1.  
* train with many epochs, particularly if you are not using a pretrained network. It may take 40-50 epochs to get a full training.  

#### train subcommand

You can train your own images with this option. Briefly, you should take light transmitted pictures of at least two conditions (control and cell death). Be aware that the more information you feed to the algorythm, the better the ability to train and predict. So, we advise that you should take at least 500 pictures in different biological replicate. Then you can slice them, and use data augmentation to increase your input.  

For image labelling, you can include in each of your image filenames either the string '*control*' or the string '*celldeath*'. These are defaults, but you can change them with the argument -labels, or even include more categories.

##### minimal example  

```bash
celldeath train -imagenet
```

with this minimal example, you just need to put your images in the folder *'~/celldeath/split_img/'* (default place if you slice them, see below), and make sure your filenames contains either *'control'* or *'celldeath'*, acording to your experiments. Defaults will probably take you to a high accuracy. We proved that our script can identify ~99% of celldeath images with minimal changes (for example, just one  hour after cell death induction). In many cases these changes are not perceptibles for the human eye. The *-pretrained* option allows you to use a neural network previously trained (with *imagenet*), which may allow to reach a high accuracy in a shorter time. However, in our experience it may not be superior to a plain training, and even a little bit inferior.

##### extended example (defaults are shown)

```bash
celldeath train -indir /your/path/img -labels yourlabels -model resnet50 -valid_pc 0.2 -l_lr 1e-4 -u_lr 1e-3 -aug -epochs 40 -bs 16 -dropout 0.5 -wd 0.01 -imagenet -test_path your/path/to/test/img
```

##### train options

Short explanations about these options are given below. Some of them may have a huge impact in your training; we suggest you to try small changes in each one of them in order to get your best trained model.  

command | help |suggestion
---   |  --- | ---
-h, --help |  show this help message and exit
-indir  |  Folder where images are stored. Be aware that default is with splitted images and so default is /img_split_train'
-labels | Give labels of each experimental condition. Labels should be written as in the filenames. Default values are '*control*' and '*celldeath*'.
-model   | Model used for training. Default is ResNet50. Models available are resnet34, resnet50, resnet101, and densenet121. | Give a change to a larger architecture, although it will take longer to train.
-valid_pct |   Validation percentage. Default is 0.2
-l_lr | Lower Limit for learning rate. Default is 1e-4 | You may try 1e-5 or even 1e-6
-u_lr |  Upper limit for learning rate. Default is 1e-3|  
-aug  |Add image augmentation. Default False | Always try it.  
-epochs  | Number of epochs. Default is 50. | Longer training may be beneficial if pretrained is false.
-bs |  Batch Size | Depends on the RAM of your GPU. Default is 16. 
-dropout |  Dropout to be applied. Defaults is 0.5 | Try 0.6-0.25
-wd | Default is 0.01 | Try 0.1 or 0.001
-imagenet | Define if training with imagnet pretrained weights. Default is False.
-test_path | Path where test images are located. Default is '/img_split_test'.

After training, a .txt file will be saved in a folder called *celldeath* (created in your home dir) with accuracy, precision and recall, as well as confusion matrix values. Also, a .csv file named *history*+current time will be saved with each of the the training loss and accuracy epochs values.  

#### predict subcommand  

After training, you can predict the presence of cell death in a set of images by using the *predict* option. Your images should be placed in a subfolder inside the training folder (*indir* from training subcommand).

##### example  

```bash
celldeath predict -path_pred indir/predict/your/img
```

##### predict options

command | help
---   |   ---
-h, --help   |   show this help message and exit
-path_pred |  Path where image/s to predict are stored.

#### slice subcommand

Your training and prediction may improve as the number of images that you have increases. If you set up your experiments where cells are confluent enough you may get use of this option. Slice will divide your picture into n tiles (default is 4), and hence increase the number of images. As far as slicing don't add images without cells you can increase your slicing up to 8 per image. Check you image capture software to see the output of the image size. For example, a typical size is 1920x1440 pixels, and hence we trained with 480x360 pixels images. 

It is a good practice to split your images into three sets: trainig, validation and test. Prediction on this last set is independent of training, and so recommended. To create a test set, add the -test_path option and the desired percentage of image to include (-perc_test).

##### example

```bash
celldeath slice -indir_slicing img/path/here -outdir_slicing your_path/img_split_train -n_tiles 4 -test_path your_path/img_split_test -perc_test 
```

##### slice options

command | help
---   |   ---
-h, --help   |   show this help message and exit
-indir_slicing |   Folder where images are stored.
-train_path |   Path where slice images are saved.
-n_tiles | Number of tiles that will be generated. Default is 4; allowed values are 2,4,6 and 8.
-test_path | Path where images for testing will be stored. Default is img_split_test.
-perc_test | Percentage of images that will be used for testing. (No default here!)

## Version

0.9.17

## Authors

* **Santiago Miriuka** | <sgmiriuka@gmail.com> | [GitHub](https://github.com/sgmiriuka) | [twitter](https://twitter.com/santiagomiriuka)
* **Alejandro La Greca** | <ale.lagreca@gmail.com>
* **Nelba Pérez** | <nelbap@hotmail.com> | [twitter](https://twitter.com/NelbaBio)

## Reference

https://www.biorxiv.org/content/10.1101/2020.03.22.002253v2

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Meli and Shei.
* the fastai team.
