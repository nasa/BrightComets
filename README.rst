BrightComets Module
===================

The BrightComets Module detects bright comets in fits images taken by WISE
and in the future NEOCAM. These images can be downloaded from irsa.ipac.caltech.edu.

The repository may be downloaded and easily modified to detect objects in your own fits or jpg image dataset. E.g., detecting stars, galaxies, planets, or even non-astronomical data like street lights or people.

The code handles fits and jpg images. All region files must be compatible with SAO Image DS9 and have the .reg extension. 

WARNING: The code has been tested with an anaconda3 installation of python 3.6.5. It has not been tested with other installations of python

Authors: Nathan Blair, ncblair@me.com, Joe Masiero, Emily Kramer, Jana Chelsey, Adeline Paiment

Quick Installation for Comet Detection
--------------------------------------

Use this installation if you only want to use the package for detecting comets in wise images

Preferably, do this in a python 3.6.5 virtual environment. If you have anaconda3, run:
::
    conda create -n brightcomets python=3.6.5 anaconda
    source activate brightcomets

And when you are finished using the program, you can deactivate your environment by closing the terminal window or running:
::
    source deactivate

If you ever want to delete your virtual environment:
::
    conda remove --name brightcomets --all

In a terminal window:
**WARNING: MAKE SURE YOUR PATH HAS NO SPACES OR SPECIAL CHARACTERS**
::
    git clone https://github.com/nasa/BrightComets
    pip install -e BrightComets

Detection
---------

In a python environment (you may need to run python with the command "pythonw" if you get an error saying Python is not installed as a framework):

the detector can take in 12 micrometer wavelength 
range infrared images taken by WISE/NEOWISE by specifying 
that you want to look detect in band_3 band_3

.. code:: python 

    from brightcomets import pipeline
    pipeline.detect("path/to/12um_fits_file", im_type="band_3")

or, if you have access to all 4 bands

.. code:: python 

    from brightcomets import pipeline

    # takes a list of fits files, from low ir bands to high ir bands.
    # intended for bands 2-4 (i.e., the list should include bands 1-4 or 2-4)
    pipeline.detect(["path/to/low_band_fits_file", "path/to/next_band_fits_file", 
                    "path/to/next_band_fits_file", "path/to/high_band_fits_file"],
                    im_type="composite")


Try this out on the given example fits data:

.. code:: python

    # From BrightComets
    pipeline.detect("brightcomets/WISE_data/00808a064/00808a064-w3.fits", im_type="band_3", do_show=True)
    import os
    pipeline.detect([f for f in os.listdir("brightcomets/WISE_data/00808a064") if f.endswith(".fits")], im_type="composite", do_show=True)


Long Installation for Custom Training/Retraining
------------------------------------------------

**WARNING: IMAGE AND REGION FILES POINTED AT BY THE RETRAINING SCRIPT MAY BE ALTERED OR DELETED BY THE PROGRAM. KEEP A COPY OF YOUR FILES ELSEWHERE ON YOUR COMPUTER**

**WARNING: IF YOU ALREADY INSTALLED A COPY OF BRIGHTCOMETS IN THE SHORT INSTALLATION, IT IS A GOOD IDEA TO REMOVE YOUR INSTALLATION BEFORE THE LONG INSTALLATION WITH "pip uninstall brightcomets"**

0. (Optional, but highly recommended) Activate a virtual environment to install dependencies for the project. 
    If you have conda, you can run:
    ::
        conda create -n brightcomets python=3.6.5 anaconda
        source activate brightcomets
    If you do not have conda, consult an online tutorial, but make sure that python version is 3.6.5

    When you want to stop working on this project, don't forget to run
    ::
        source deactivate

    If you ever want to delete your virtual environment:
    ::
        conda remove --name brightcomets --all


1. Get the code, install requirements. 
    In a terminal window: 
    ::
        git clone https://github.com/joemasiero/BrightComets
        cd BrightComets
        pip install -r requirements.txt


2. Install the tf object_detection library in the BrightComets directory
    First, go to https://github.com/google/protobuf/releases and download protobuf-all-3.6.1.tar.gz
    ::
        # From the location where you downloaded protobuf (possibly Downloads)
        tar -xvf protobuf-all-3.6.1.tar.gz
        cd protobuf-3.6.1
        ./configure
        # This may take a while
        make
        sudo make install
        protoc --version # check installation worked
    Then, run the following commands:
    ::
        # From path/to/BrightComets
        git clone https://github.com/tensorflow/models.git
        cd models
        git checkout 3a05570f8d5845a4d56a078db8c32fc82465197f
        cd ..
        git clone https://github.com/cocodataset/cocoapi.git
        cd cocoapi/PythonAPI
        make
        cp -r pycocotools ../../models/research
        cd ../../models/research
        protoc object_detection/protos/*.proto --python_out=.

    More info about this step can be found here_:

    .. _here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

3. Run the file_organization script. 
    This will make some compatibility changes to the object_detection library and also add a line to your ~/.bash_profile file so that the object_detection library can be properly imported. 
    ::
        # From BrightComets/brightcomets
        # cd ../../brightcomets
        python file_organization.py
    If this command gives you an error, you may not have a ~/.bash_profile file. And you will have to manually type the following line whenever you open a new terminal window and want to run the retraining script.
    ::
        # If the previous command gave an error
        # From BrightComets/models/research/
        export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    Now, the library should be installed and prepared for retraining. 


4. Download `SAO Image DS9`_ if you do not already have it

    .. _SAO Image DS9: http://ds9.si.edu/site/Download.html

5. Compile training data
    **WARNING: IMAGE AND REGION FILES POINTED AT BY THE RETRAINING SCRIPT MAY BE ALTERED OR DELETED BY THE PROGRAM. KEEP A COPY OF YOUR FILES ELSEWHERE ON YOUR COMPUTER**

    Change to your BrightComets/brightcomets directory

    Compile a folder with .fits and .reg files in it. For example, take a look at the WISE_data folder. The files don't need to be structured in any specific way, however matching fits and regions files should have the same name.

    File naming conventions:

    A. Image files may either be of type .jpg or .fits. All .jpg files will be converted to .fits. 

    B. Image files will be resized to the shape 512 by 512. If they are not 512x512, they will be stretched to 512x512. 

    C. All image files should have unique names. 

    D. Regions are specified as .reg files and use the syntax conventions from SAO Image DS9 region files. They must have the .reg file extension. 

    E. All region files should correspond to a fits file by having the same name as that image file. 

    F. The dash ("-") character is an important keyword
        i. They allow the user to specify that two images represent different channels of the same image. 

        ii. For example, the files image1-w1, image1-w2, and image1-w3 will all be placed in the image1 folder, and can be used to make a composite image with w1 representing the red channel, w2 representing the green channel, and w3 representing the blue channel. 

        iii. The program supports up to 4 channels of the same image. They should be named w1, w2, w3, and w4. 

        iv. When training, you will have the option to train on all channels separately, just channel 3 (w3), or a composite of the three highest channels. 

        v. If there are two dash ("-") characters in the name of a file, the second dash and everything after it will be truncated. i.e. image1-w1-int1-abc.fits will be truncated to image1-w1 and placed into the w1 folder

        vi. A region file corresponding to the composite of images in a folder will have the extension -comp. For example, a file may be named image1-comp.reg

        vii. Be very careful when having dashes in your file names. They should only be used when followed by either w1, w2, w3, w4 or comp. And, they will signify that the images are different channels of the same image. 

        viii. When region and fits files are matched into folders, they are matched only by the uniqueness of the name before the first dash. Once the files have been organized, regions and fits files of corresponding channel will be matched during runtime. For example, image1-w2.fits will be matched with image1-w2.reg

    **What if I don't have region files?**

    You will be prompted upon running the retraining script to create your own regions

    **What if I have region files without corresponding fits files?**

    Those region files will be deleted. Make sure to keep a copy elsewhere on your computer! 

6. Use config.py to set hyperparameters for training. 
    A. Open BrightComets/brightcomets/config.py

    B. Open SAO Image DS9, and keep it open for the rest of the training process.

    C. Go to File > XPA > Information

    D. In the config.py file, copy the text after XPA_METHOD and set the variable FITS_XPA_METHOD = "YOUR_XPA_METHOD"

    E. In the config.py file, set pyversion to the command that you type in the terminal to invoke python 3.6.5. This is probably just "python", but could also be "pythonw" or "python3", for example. This is necessary because the program makes system calls to run the training and evaluation scripts. 

    F. If you are using custom training data, set color_key to a dictionary describing how your annotations are labelled, or how you wish them to be labelled if they are not yet labelled. Use default_color_key as a reference. 

    G. Default image resizing is to 512x512. If you change the image_size parameter here, it will be changed everywhere for all networks. It is recommended that you do not change this parameter, as changing the image size has not been heavily tested. 

7. Run the retraining script
    In a terminal window: 

    This will display all the retraining options you have
    ::
        # From BrightComets/brightcomets
        python retrain.py -h

    If you get an error saying python is not installed as a framework, try
    ::
        # replace all future calls to python with pythonw
        pythonw retrain.py -h

        # Also, change your pyversion variable in the config.py file. 

    The command you will run will look something like
    ::
        # From BrightComets/brightcomets
        # Make sure to replace Your_Custom_Data folder with the path to your folder
        python retrain.py --im_type band_3 --update_records --data_folders Your_Custom_Data --retrain --train_iterations 5000 --classes comet

    This script does training, evaluation, and metrics all at once. 

    You can see the training and evaluation progress by going to a web browser while the script is running and searching **localhost:6006**

8. Once you are satisfied with your model, you can download it. 
    From the BrightComets directory
    ::
        # Inside the BrightComets Directory
        pip install -e .
    This will allow it be available everywhere on your computer. If you intend to make more changes locally. I recommend uninstalling it till you are finished making changes. Otherwise, you will have to update your installation every time you make changes for testing. 
    Uninstall with:
    ::
        pip uninstall brightcomets
    This will not get rid of your local copy of the repository. 

Objects with comets already annotated (and some stars, planets, and defects annotated):
These annotations are stored in regions-static and will always be checked. However, annotations given by the user will always be prioritized. 
comets = ["C/2006 W3", "C/2007 Q3", "65P", "29P", "30P", "81P", "116P", "10P", "118P", "P/2010 H2", "C/2007 N3"]
not_comets = ["mars", "jupiter", "alpha boo", "R dor"]

FAQ
===

**Can you train multiple different kinds of models?**

Yes! You can, for example a model that looks at band_3 stars, a model that looks at band_3 comets, a model that looks at composite comets. You cannot train two different band_3 comet models however. 

Note that the config.py file is global to all models, while each call to retrain.py will only retrain a single model. The retrain script may be slow as it preprocesses the images every time it is run.

**Will this program run on all operating systems?**

This module is built for mac. I make no gurantees that it will work on other operating systems. 

**My program isn't working. What do I do?**

Are you using the right version of python (3.6.5)? Do you have all necessary libraries installed (if not, install with pip)? Do you have your XPA_METHOD and pyversion properly set in your config file? Are your data files properly named, with dashes in the correct places? Did you reinstall the library with pip after making changes (You have to reinstall it every time you make changes locally). Make sure none of your filepaths have spaces or special characters like backslashes. 


Brief Folder and File Descriptions
==================================

1. BrightComets/references.md
    Some websites I referenced during the creation of the project, some comments point to these references. 
2. BrightComets/requirements.txt
    All dependencies. Install requirements with pip install -r requirements.txt
3. BrightComets/setup.py
    File that allows the project to be installed via pip. pip install -e BrightComets
4. BrightComets/brightcomets
    a. data
        Where tensorflow records and label files are stored for all neural networks. These files are binary files, not human readable, but store all fits and regions files after preprocessing. 
    b. master_data
        Fits and Regions files organized into test and train datasets
    c. models
        Tensorflow object detection models
        Check here for more models: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
        This is where all tensorflow computational graphs and weights are stored for each model you train. 
    d. regions-static
        All the regions that are provided upfront
    e. unused
        Files that I did not end up using but still have interesting code in them. The files are mostly intended to skirt os.system calls in the retraining script
    f. WISE_data
        An example of how training data can be organized.
    g. __init__.py
        A short file for making BrightComets a python package
    h. config.py
        Configuration file where the user sets training hyperparameters
    i. eval.py
        (Created by Google Tensorflow) Tensorflow object_detection file that calls does evaluation
    j. export_inference_graph.py
        (Created by Google Tensorflow) Tensorflow object_detection file that exports trained neural nets
    k. file_organization.py
        File that does all organization of training data, file movement, TFRecord creation, etc.
    l. image_methods.py
        File that has the main object detection algorithm, preprocessing algorithms, and fits/regions handling algorithms. 
    m. pipeline.py
        File that allows user to use the detection algorithm
    n. retrain.py
        File that allows the user to retrain and organize/annotate training data
    o. tests.py
        Mostly deprecated file with some tests, mostly for image_methods functions
    p. train.py
        (Created by Google Tensorflow) Tensorflow object_detection file that initiates neural network training, called by retrain.py
    q. utils.py
        static methods and utils
