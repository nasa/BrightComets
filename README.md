# BrightComets

## Instructions:

- clone or download the repo, change directories into .../BrightComets
- [optional] Create and activate a virtual environment using python3
- Install dependencies:
```
pip install -r requirements.txt
```
- Make sure the WISE_Comets folder is in your directory (Email Nathan if you need the master)
- Change the file structure of the WISE_Comets folder (keep the zip file if you want to revert)
```
python organize_folders.py
```
- Open SAO DS9
- FITS_XPA_METHOD needs to be set at the top of image_methods.py
	- To find the correct value: in your version of ds9:
		- File > XPA > Information > XPA METHOD

	- Note: You only need to 

- to make sure everything works on your computer (may take around 15 minutes):
```
python tests.py --quiet
```
- image_methods.py is most important file, has all image processing and detection stuff
- move_ims_to_tf_loc.py moves images to the folder where the tensorflow files are
- randomize_names.py randomizes the names in that ^ folder
- organize_folders.py takes photos from Wise Comets and moves them and reorganizes them
- utils.py has utility functions
- You may also want to look at tests.py for examples of how the code works

- To detect whether an image numpy array has a comet in it, run:
```
# if you would like to display the process in an interactive window:
image_methods.detect_comet(imarray, do_show=True)
# otherwise: 
image_methods.detect_comet(imarray)
# if you are getting unexpected errors, try seeing if it works with parallelize=False
```

## Retraining
Standardize Input (master_data folder):

This is the format that will be turned into a TFRecord file, as specified in [5]. 
So, A TFRecord file is also acceptable as training input. 

Once it is a TFRecord file, it will be shuffled and split up into train, 
validation, and test sets. There will also be an option to reshuffle. 

master_data:
	sub_folder:
		sub_folder-w{band}.fits x {num_bands}
		AND
		sub_folder-w{band}.reg x {num_bands}

Protocol: 
- If one of the bands is corrupted, replace it with the nearest (default up) uncorrupted band 
	- (e.g., replace corrupted band 1 with band 2, replace corrupted band 4 with band 3, corrupted band 3 with band 4).
- If we want to use multiple bands for a single image, default to top 3 bands as rbg


- If no region file for corresponding fits file:
	- you will automatically be prompted to annotate the image

Region File Specifications:
	The region files should be astropy region files with image pixel coordinates
	The color of each region should be:
	color_key = {
		"comet" : "green",
		"one_star" : "white", 
		"mul_star" : "red", 
		"defect" : "blue"
	}
	Importantly, the program assumes that there will only be a single comet per image, 
	and thus merges all green regions into a single region surrounding the originals
	For non-comet regions however, it is assumed that every shape (e.g. box) in the 
	region file corresponds to a unique region containing a unique object. 


For Example: 
	```
	# Region file format: DS9 version 4.1
	global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
	image
	circle(359.00075,124.99951,5.8728111)
	ellipse(377.00068,127.00032,23.000058,10.999998,359.99993)
	ellipse(394.99898,164.00005,0,0,359.99993)
	# vector(360.99902,127.00029,148.75824,5.7869338) vector=1
	# vector(362.00057,125.99954,300.54045,187.45593) vector=1
	box(194.49966,130.50082,17.000005,20.999967,359.99993) # color=white
	box(192.49952,294.99909,32.999966,29.999979,359.99993) # color=blue
	box(32.000198,164.00068,13.999997,13.999996,359.99993) # color=blue
	box(13.999095,199.99961,7.9999983,16.000001,359.99993) # color=blue
	box(114.00081,394.99999,21.999977,20.000046,359.99993) # color=red
	```