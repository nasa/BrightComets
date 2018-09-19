try:
	from brightcomets.image_methods import detect_comet, preprocess, get_image_data
except ModuleNotFoundError:
	print("Assuming you are running locally")
	from image_methods import detect_comet, preprocess, get_image_data

import numpy as np

def detect(fits_data, im_type="all_bands",
			model="faster_rcnn_inception_v2_coco", classes=["comet"],
			do_show="False"):
	"""
	fits_data (list or string) if list, list of fits files.
			If string, path to single fits file. 
	im_type (string) band_3, all_bands, or composite. What
			type of neural net to use for detection
	model (string) directory in models. Probably don't want
			to change this. can download additional models 
			from tf object detection model zoo
	classes: Which classes the neural net you want to use is
			trained on. For example, if your neural net was 
			trained on classes named "roses" and "dandelions",
			classes would look like: ["roses", "dandelions"]
	do_show: Whether or not to show the image in a matplotlib 
			popup window
	---
	RETURNS:
		imarray (array) the image, preprocessed and resized
		boxes (list) list of bounding box coordinates
		scores (list) list of confidences for each bounding box
		classes (list) list of classes that each bounding box was classified as
	"""
	classes_string = "".join([x[0] for x in classes])
	if isinstance(fits_data, list):
		images = [get_image_data(f)[0] for f in fits_data]
	else:
		images = [get_image_data(fits_data)[0]]
	images = [preprocess(im, resize=False) for im in images]

	imarray, boxes, scores, classes = detect_comet(
					images, do_show=do_show, 
					im_type=im_type, 
					model=model, 
					classes_string=classes_string)
	return imarray, boxes, scores, classes