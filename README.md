# **celldeath**

A simple python script based on deep learning optimized for classifying cell death in transmitted light microscopy images. 
<img src="./img.png" align="right" height="200" width="200" >

Our goal is to provide a tool for researchers with none  or minimal idea of deep learning.

&nbsp;
## Getting Started

celldeath is a deep learning tool that trains with light transmitted microscopy images in order to predict cell death/apoptosis.

We aimed to keep it simple, so anyone can implement it with minimal deep learning knowledge. Parameters for training are already optimized for high accuracy. However, we provide some suggestions for fine tunning your training, as each experimental model may have unique features.

&nbsp;
### Prerequisites

Please note that for training a deep learning model you need a GPU and lots of images. If you don't have a GPU available, you can try in any cloud computing provider (AWS, Google Cloud, Azure, etc).

&nbsp;
### Installing

For installation, we suggest that you create an environment, and then:

```bash
pip install celldeath
```
&nbsp;
### Dependencies

* fastai>=1.0.60
* image-slicer==0.3.0
* matplotlib>=3.1.1
* torch==1.4
* torchvision==0.5.0

&nbsp;
## Usage

celldeath has three modules (*train*, *predict* and *slice*), each one with options of their own (see section Modules for details). However, for a shorter version, you can follow these next steps using default parameters. 

#### Labelling
  
For image labelling, you can include in each of your image filenames either the string '*control*' or the string '*celldeath*'. These are defaults, but you can change them with the argument -labels, or even include more categories.

```bash
# possible naming formats
testFile_control_RGB_1h.png
testFile_celldeath_RGB_1h.png
```

&nbsp;
### Short version

Open your terminal and type

```bash
celldeath slice -indir_slicing your/img/folder # path to your input images; output is directed by default to img_split_train, unless -train_path argument is provided
```

> Be aware that if you train several times with the same images, this previous step has to be done just once. This command will slice your images into 4 tiles, which will be used later for training.

```bash
celldeath train -labels your_labels -imagenet # example for label input: -lables control celldeath
```

You have to provide your labels, which must be included as part of the filename in each image. Default values are *control* and *celldeath*. You can have more than 2 conditions though. 

Default values should lead to high accuracy in your images for prediction. (Of course, your images should be good enough...) Once training is finished, you will find a report file in ~/celldeah directory that includes all training parameters, results on the validation image set, and accuracy on the test set if performed. You will also get a confusion matrix with absolute numbers for prediction and the Loss curve for training and validation.  

&nbsp;
## Modules

The next subcommands will give you some more control on preprocessing, training and prediction. Subcommands are:

* **celldeath slice** | Only need to run once, and only if you wish to slice your images.

* **celldeath train** | Core subcommand for training the neural network.

* **celldeath predict** | One or more images are given and it returns prediction based on your previous training.  

&nbsp;
### SLICE

Your training and prediction may improve as the number of images that you provide increases. If cells were homogeneously seeded in your experiments you may find this module useful. Slice will divide your picture into n tiles (default is 4), and hence increase the number of images. As long as the slicing process does not add images without cells, you can increase your slicing up to 8 per image. For example, a typical image size is 1920x1440 pixels, and hence after slicing into 4 tiles, we trained with 480x360 pixels images. 

Also, it is a good practice to split your images into three sets: trainig, validation and test. Prediction on this last set is independent of training, and so recommended. To create a test set, add the -test_path option and the desired percentage of image to include (-perc_test).

#### Example

```bash
celldeath slice -indir_slicing path/to/input -train_path ~/img_split_train -n_tiles 4 -test_path your_path/img_split_test -perc_test 
```

### Slice options

command | description
---   |   ---
-h, --help   |   show this help message and exit
-indir_slicing |   Folder where images are stored.
-train_path |   Path where sliced images are saved.
-n_tiles | Number of tiles to generate. Default is 4; allowed values are 2,4,6 and 8.
-test_path | Path where images for testing will be stored. Default is img_split_test.
-perc_test | Percentage of images that will be used for testing. (No default here!)

&nbsp;

### TRAIN

celldeath allows you to train your own images without too much knowledge on deep learning. Be aware though, that you need at least a few hundred images for proper training and prediction. However, as mentioned before, with the *slice* module you can split your images into n tiles (usually 4) increasing your training performance. Also, using the argument -aug will artificially increase the information gathered from each image by flipping and rotating them. The *train* module already has most options set up for an excellent training, as mentioned in the short version section.

&nbsp;
### Minimal example with pretrained models (imagenet) 

```bash
celldeath train -imagenet
```

In this minimal example, you just need to put your images in *'~/celldeath/img_split_train/'* (default path if you used the slice module, see SLICE section), and make sure your filenames contain the labels *'control'* or *'celldeath'*, acording to your experiments. The *-imagenet* option allows you to use a neural network previously trained (with *imagenet*), which may result in a accurate short-running-time training. However, in our experience it did not produce better results (lower test accuracy).

&nbsp;
### Extended example (defaults are shown)

```bash
celldeath train -indir /your/path/img -labels control celldeath -model resnet50 -valid_pc 0.2 -l_lr 1e-4 -u_lr 1e-3 -aug -epochs 40 -bs 16 -dropout 0.5 -wd 0.01 -imagenet -test_path your/path/to/test/img # 'control' and 'celldeath' labels are shown, but you may use your own as long as they can be found somewhere in the filename
```
&nbsp;
### Train options

Short explanations about these options are given below. Some of them may have a huge impact in your training; we suggest you to try small changes in each one of them in order to get your best trained model.  

command | description |suggestion
---   |  --- | ---
-h, --help |  show this help message and exit
-indir  |  Folder where input images are stored. Default is '/img_split_train' (resulting from slice module)
-labels | Give labels of each experimental condition. Labels should be written as in the filenames. Default values are '*control*' and '*celldeath*'.
-model   | Model used for training. Default is ResNet50. Models available are resnet34, resnet50, resnet101, and densenet121. | Trying larger architectures will take longer to train.
-valid_pct |   Validation percentage. Default is 0.2
-l_lr | Lower Limit for learning rate. Default is 1e-4 | You may try 1e-5 or even 1e-6
-u_lr |  Upper limit for learning rate. Default is 1e-3|  
-aug  |Add image augmentation. Default False | Always try it.  
-epochs  | Number of epochs. Default is 50. | Longer training may be beneficial to detect porential overfitting if imagenet option is false.
-bs |  Batch Size | Depends on the RAM of your GPU. Default is 16. 
-dropout |  Dropout to be applied. Defaults is 0.5 | Try 0.6-0.25
-wd | Default is 0.01 | Try 0.1 or 0.001
-imagenet | Define if training with imagnet pretrained weights. Default is False.
-test_path | Path where test input images are located. Default is '/img_split_test'.

&nbsp;

After training, a .txt file will be saved in a folder called *celldeath* (created in your home dir) with accuracy, precision and recall, as well as confusion matrix values. Also, a .csv file named *history*+current time will be saved with each of the the training loss and accuracy epochs values.  

&nbsp;
### A few recommendations:  

* be consistent in your experimental conditions. 
* get as many images as possible all over the well.
* try both pretrained (-imagenet) and not pretrained. Pretrainng may be faster, but it is not necessarily better though (there are no cells in imagenet).  
* try augmentation. We set up a few augmentation parameters (flip and rotation, minimal changes in light condition) since some other common arguments result in a lower performance.  
* batch size (-bs) will depend on your GPU and the size of your images.  
* Always try weight decay. We set it up to 0.01, but also try 0.1.  
* train with many epochs, particularly if you are not using a pretrained network. It may take 40-50 epochs to achieved a fully trained model.

&nbsp;

### PREDICT 

After training, you can predict the presence of cell death in a set of images by using the *predict* option. Your images should be placed in a subfolder inside the training folder (*indir* from training subcommand).

#### Example  

```bash
celldeath predict -path_pred indir/predict/your/img
```

### Predict options

command | description
---   |   ---
-h, --help   |   show this help message and exit
-path_pred |  Path where image/s to predict are stored.

&nbsp;
## Version

0.9.17

&nbsp;
## Authors

* **Santiago Miriuka** | <sgmiriuka@gmail.com> | [GitHub](https://github.com/sgmiriuka) | [twitter](https://twitter.com/santiagomiriuka)
* **Alejandro La Greca** | <ale.lagreca@gmail.com> | [GitHub](https://github.com/alelagreca) | [twitter](https://twitter.com/aled_lg)
* **Nelba Pérez** | <nelbap@hotmail.com> | [GitHub](https://github.com/nelbaperez) | [twitter](https://twitter.com/NelbaBio)

## More info and citation

https://www.biorxiv.org/content/10.1101/2020.03.22.002253v2

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Meli and Shei.
* the fastai team.
