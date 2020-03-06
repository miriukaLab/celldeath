<a href="url"><img src="./img.png" align="right" height="250" width="250" ></a>

## **celldeath**

### *A simple python script based on deep learning for classifying cell death in cell culture*



## Getting Started 
 
CellDeath is a simple deep learning script that trains with light transmitted microscopy
images and then predict if those images contains cells undergoing cell death/apoptosis.

Subcommands are:

> **slice** | Only needed to run once, and only if you need to slice your images. 

> **train** | Core subcommand for training the neural network. 
    
> **predict** | One or more images are given and it returns if those cells are undergoing cell death.  

We provide a pretrained model. You can predict your images with it, although results are far to be far to be accurate, unless you have taken images as we did: 10x objective, images sliced in four, final shape 360x480 pixels, and may be, have used any of the cell lines we used.  So, we suggest to train your model with your own images. 

### Prerequisites

Please note that for training a deep learning model, you need a GPU and a lot of images.   

### Installing

For installation, just pip it. 

```
$ pip install celldeath
```

### Dependencies


* fastai 1.4
* image-slicer 
* PIL


## Usage

celldeath has three subcommands (*train*, *predict* and *slice*), each one with their options. 
 
### train

celldeath allows you to train your own images without too much knowledge of deep learning. Be aware that you may need a few hundreds of images at least for proper training. However, with the subcommand *slice* you can split your images in n tiles (usually 4), and hence increase your training performance. The subcommand *train* already has most of the option set up for a decent training, and so you can provide a minimal input (just the path where your images are stored) to get a high accuracy, providing that your images and your experiement are reasonable.

A few recommendations: 
* get as many images as possible.
* try both pretrained (imagenet) and not pretrained. Pretrainng is not necessary better. 
* try augmentation. We set up a minimal augmentation (flip and rotation, minimal changes in light) since too many arguments for this results in a lower performance. 
* batch size (-bs) will depend on you GPU. A decent RAM (11Gb) should accept a batch size of 32-64, depending on the size of your images. 
* Always try weight decay.
* try many epochs, particularly if you are not using a pretrained network. It may take 40-50 epochs to get a full training. 




#### train options

You can train your own images own images with this poption. Briefly, you should take light transmitted pictures of at least two conditions (control and cell death). Be aware that the more information you feed to the algorythm, the better the ability to train and predict. So, we advise that you should take at least 500 pictures in different biological replicate. Then you can slice them, and use data augmentation to increase you input. 

For image labelling, you ***must*** include in your image filenames either the string '***control***' or the string '***celldeath***'. 

minimal example 
```
$ python main.py train -pretrained
```
with this mininmal example, you just need to provide the path to your images. Defaults will probably take you to a high accuracy. The -pretrained option allows you to use a neural network previously trained (with *imagenet*), which may allow to reach a high accuracy in a shorter time. However, in our experience it is not superior to a plain training, and even a little bit inferior.  

extended example (defaults are shown)

```
$ python main.py train -indir ~/split_img -model resnet50 -valid_pc 0.2 -l_lr 1e-4 -u_lr 1e-3 -aug -epochs 40 -bs 16 -droput 0.5 -wd 0.1 -pretrained 
```
Short explanations about these options are given below. Some of them may have a huge imact in your training; we suggest you to try small changes in each one of them in order to get your bets trained model. 

command | help |suggestion
---   |  --- | ---
-h, --help |  show this help message and exit
-indir  |  Folder where images are stored. Beaware that default is with splitted images and so default is /split_img
-model   | Model used for training. Default is ResNet50. Models availbe are resnet34, resnet50, resnet101, and densenet121. | Give a change to deeper models, although it will take longer to train.
-valid_pct |   Validation percentage. Default is 0.2
-l_lr | Lower Limit for learning rate. | You may try 1e-5 or even 1e-6
-u_lr |  Upper limit for learning rate. | 
-aug  |Add image augmentation. Default False | Always try it. 
-epochs  | Number of epochs. Default is 30. | Longer training may be beneficial if pretrained is false.
-bs |  Batch Size | Depends on your GPU.
-dropout |  Drop out to be applied. | Try 0.4, or even 0.3
-wd | Default is 0.1 | Try 0.01
-pretrained | Define if train using Imganet pretrained weights. Default is False.

#### predict options 

If you have your own set of images you can try this subcommand (with the -pretrained option) and find out what the accuracy is on it. We provide a pretraind model, based on a training on 7 cell lines exposed to one drug for one hour.   

Beaware that chances of a good performance will depend on many variables, and eventually it may be useless. It is much better to train your network and then predict on subsecuent experiments (that is, without -pretrained option). In this case, this subcommand will lok for the last trained NN and predict based on it. 

example 

````
$ python main.py predict -path_pred img/path/here -pretrained 
````
command | help
---   |   ---
-h, --help   |   show this help message and exit
-path_pred |  Path where image/s to predict are stored.
-pretrained   |   Use provided pretrained model. If used, you should have had trained with your images before.

#### slice options

example 
```
$ python main.py slice -indir_slicing img/path/here -outdir_slicing img/path/here -n_tiles 4
```

command | help
---   |   ---
-h, --help   |   show this help message and exit
-indir_slicing |   Folder where images are stored.
-outdir_slicing |   Folder where slice images are saved. Default is ~/celldeath/split_img
-n_tiles | Number of tiles that will be generated. Default is 4; allowed values are 2,4,6 and 8.

## Version

1.0.0

## Authors

* **Santiago Miriuka** - <sgmiriuka@gmail.com> | [GitHub](https://github.com/sgmiriuka) | [twitter](https://twitter.com/santiagomiriuka)
* **Alejandro La Greca** <ale.lagreca@gmail.com>
* **Nelba Pérez** <nelbap@hotmail.com>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Meli and Shei. 
* the fastai team. you make us possible. 

