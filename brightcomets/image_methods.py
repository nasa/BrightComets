####
# Imports
####

import math
import multiprocessing as mp
import os
import time
from functools import reduce

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pyds9
import pyregion
import resource
import scipy.ndimage
import scipy.optimize
import scipy.signal
import skimage.transform
import tensorflow as tf
from astropy.io import fits
from matplotlib.colors import LogNorm
from PIL import Image

try:
	from brightcomets import config, utils
except ModuleNotFoundError:
	print("Assuming you are running locally")
	import config, utils

IMSIZE = config.im_size
####
# Globals
####

# Set this to Connect to DS9
# in your version of ds9: go to File > XPA > Information > XPA METHOD 
FITS_XPA_METHOD = config.FITS_XPA_METHOD

# override print function
print = utils.special_print(verbose=True)

# Allow your machine to open enough files to process the data
# This is necessary because the pyregion library does not properly close files
# And, since I have no way of referencing their opened files, I am stuck. 
resource.setrlimit(resource.RLIMIT_NOFILE, (2000, 2000))

# If you get a too many open files warning:
# Try running ulimit -n 1000 in your terminal window. This will increase the
# number of files that you will be allowed to open.
print("\nYou may encounter problems training on more files than:")
os.system("ulimit -n")
print("\n\n")

####
# Functions for comet detection
####

@utils.timeit
def detect_comet(imarray, do_show=True, im_type="all_bands",
						model="faster_rcnn_inception_v2_coco", 
						classes_string="c"):
	"""
	TAKES PREPROCESSED IMAGES
	imarray can be of type: 
		(list of 2d np.arrays, 2d np.array, 3d np.array or 4d np.array)
	if imarray is a list of np.arrays:
		they will each be preprocessed, and made a composite image
	if imarray is a 2d np.array:
		it will be preprocessed, and resized appropriately
	if imarray is a 3d np.array (tensor):
		it will have an extra dimension added
	if imarray is a 4d np.array (tensor):
		it will do nothing to preprocess the image
	"""
	path_to_model = f"{utils.base_dir()}/models/{model}/{im_type}/"\
					+ f"{classes_string}/tuned/frozen_inference_graph.pb"
	path_to_labels = f"{utils.base_dir()}/data/{im_type}/{classes_string}/"\
					+ "train/labels.pbtxt"
	# normalize imarray:
	if isinstance(imarray, list) or imarray.ndim != 4:
		if isinstance(imarray, list):
			imarray = [resize_image(im, IMSIZE) for im in imarray]
			imarray = get_composite_image(imarray)
		elif imarray.ndim == 2:
			imarray = get_composite_image([imarray])
		imarray = read_tensor_from_image_array(imarray, 
			input_height=IMSIZE, 
			input_width=IMSIZE, 
			normalize=False)


	detection_graph = load_graph(path_to_model)
	with detection_graph.as_default():
		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
		d_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
		d_scores = detection_graph.get_tensor_by_name('detection_scores:0')
		d_classes = detection_graph.get_tensor_by_name('detection_classes:0')
		num_d = detection_graph.get_tensor_by_name('num_detections:0')

	with tf.Session(graph=detection_graph) as sess:
		boxes, scores, classes, num = [e[0] for e in sess.run(
			[d_boxes, d_scores, d_classes, num_d], 
			feed_dict={image_tensor: imarray})]

	imarray = np.squeeze(imarray, axis=0)

	boxes[:, :2] *= imarray.shape[0]
	boxes[:, 2:] *= imarray.shape[1]
	t = np.sum(scores > .33)
	boxes, scores, classes = boxes[:t], scores[:t], classes[:t]

	labels = [x for x in config.color_key]
	classes = [labels[int(i) - 1] for i in classes]
	print([(a, b, c) for a, b, c in zip(boxes, scores, classes)])
	if do_show == True:
		show(imarray, use_log=False, annotations=zip(boxes, classes))
		if "comet" in classes:
			i = classes.index("comet")
			boxes, classes = [boxes[i]], [classes[i]]
			show(imarray, use_log=False, annotations=zip(boxes, classes))
	return imarray, boxes, scores, classes




###
# Functions for tensorflow interfacing
###

def read_tensor_from_image_array(imarray, input_height=299, input_width=299, 
									nested=True, normalize=True):
	"""
	IMARRAY: (2d np.array) numpy array, grayscale
	---
	RETURNS: (tf.Tensor) rgb image representation of numpy array
	---
	REFERENCE: [4]
	"""
	input_mean = np.mean(imarray)
	input_std = np.std(imarray)
	im_tens = tf.convert_to_tensor(imarray)
	im_tens = tf.cast(im_tens, tf.uint8)
	im_tens = tf.expand_dims(im_tens, 0);
	im_tens = tf.image.resize_bilinear(im_tens, [input_height, input_width])
	if not nested:
		im_tens = tf.squeeze(im_tens, axis=0)
	if normalize:
		im_tens = tf.divide(tf.subtract(imt_tens, [input_mean]), [input_std])
	with tf.Session() as sess:
		result = sess.run(im_tens)

	return result


def load_graph(model_file):
	"""
	MODEL_FILE: (str) path to file with model
	---
	RETURNS: (tf.Graph) graph object loaded from file
	---
	REFERENCE: [4]
	"""
	graph = tf.Graph()
	with graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(model_file, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')
			return graph

####
# Functions for Fits and Regions handling
####

def get_image_data(fits_file):
	"""
	FITS_FILE: (str) path to fits file
	---
	RETURNS:    (2d np.array) image array
				(fits header) image header
	"""
	assert fits_file.endswith(".fits") and os.path.exists(fits_file), \
		f"file: {fits_file} must be a valid absolute path to a .fits file"
	header = fits.getheader(fits_file)
	imarray = fits.getdata(fits_file)

	return imarray, header

def get_region_masks(fits_file, region_file, mode="comet"):
	"""
	FITS_FILE: (str) path to fits file
	REGION_FILE: (str) path to region file
	MODE: (str) in ["comet", "one_star", "mul_star", "defect", "labels"]
				if mode == labels, return all masks and a list of labels
	---
	RETURNS: (list) region masks for each object of type MODE
	"""

	if mode == "labels":
		label_key = dict((reversed(item) for item in config.color_key.items()))
		colors = list(label_key)
	else:
		colors = [color_key[mode]]

	region_string = utils.get_text_in_file(region_file)

	with fits.open(fits_file) as f:

		# pyregion objects are used to get region masks
		if any([not (x.startswith("#") or x.startswith("global") \
			or x.startswith("image")) for x in region_string.split("\n")]):
			s_coord = pyregion.parse(region_string)

			# get all regions relevarnt to the mode
			new_r_coord_list = [s for s in s_coord if s.attr[1]["color"] \
								in colors]


			#get ordered labels of bounding boxes
			if mode == "labels":
				labels = [label_key[s.attr[1]["color"]] for s \
							in new_r_coord_list]

			# get a pyregion object from a python list
			r_coord = pyregion.ShapeList(new_r_coord_list)
		else:
			r_coord = []
			if mode == "labels":
				labels = []

		# get the masks that the function returns
		masks = []
		for shape in r_coord:
			object_r_coord = pyregion.ShapeList([shape])

			mask = object_r_coord.get_mask(hdu=f[0])

			# pyregion doesn't convert vectors to a mask, so I add a point in
			# the mask for each of the endpoints of the vector manually
			if shape.name == "vector":

				# get vector info
				x, y, length, angle = [int(x) for x in shape.coord_list]
				angle = angle * math.pi / 180

				# get vector length and endpoints, watch out for borders
				xlength = length * math.cos(angle)
				ylength = length * math.sin(angle)
				xendpoint = min(int(x + xlength), mask.shape[0]-1)
				yendpoint = min(int(y + ylength), mask.shape[1]-1)
				xendpoint, yendpoint = max(xendpoint, 0), max(yendpoint, 0)

				# modify mask to include vector start and endpoints
				mask[y, x], mask[yendpoint, xendpoint] = [True] * 2

			masks.append(mask)

	# if we are looking for a comet, each region should be combined to make one
	# This is because a comet is made up of a vector, ellipse, and circle
	# we assume one comet per image
	if mode == "comet":
		masks = [reduce(np.logical_or, masks)]

	if mode == "labels":
		# merge comets to one region, "only one comet per image"
		c = [m for m, l in zip(masks, labels) if l == "comet"]
		c = [reduce(np.logical_or, c)] if c != [] else []
		ms = [m for m, l in zip(masks, labels) if l != "comet"] + c
		ls = [l for l in labels if l != "comet"] + ["comet"] * (c != [])
		return ms, ls
	return masks

def convert_jpg_to_fits(jpg_path):
	im = scipy.ndimage.imread(jpg_path, mode='L')
	im = resize_image(im, max(im.shape))
	f = fits.PrimaryHDU(data=im)
	f.writeto(f"{jpg_path[:-4]}.fits")
	os.remove(jpg_path)

def annotate_image(fits_file, reg_file=None):
	print(f"You are going to give an annotation for {fits_file}")
	input("You will be prompted from the command line, keep it open \n\
			Press enter to continue")
	with fits.open(fits_file) as f:
		try:
			dim=pyds9.DS9(FITS_XPA_METHOD)
		except:
			print("Looks like your SAO DS9 is not connected")
			xpa_method = input("Please input you XPA METHOD \n\
					This can be found in your version of ds9:\n\
					\tFile > XPA > Information > XPA METHOD")
			dim = pyds9.DS9(xpa_method)

		# get the pixel-image coordinate representation of region 
		outf=fits.HDUList(f[0])
		dim.set("frame new")
		showtest=dim.set_pyfits(outf)
		if reg_file is not None:
			region_string = utils.get_text_in_file(reg_file)
			dim.set('regions', region_string)
		dim.set("regions -system image")
		dim.set("regions shape box")

		for key, val in config.color_key.items():
			dim.set(f"regions color {val}")
			print(f"Now, annotate all {key}")
			input("When you are done, press return")

		region_string = dim.get("regions -system image")
		dim.set("frame delete")
	return region_string

@utils.timeit
def annotate_composite(fits_files, reg_file=None, size=IMSIZE):
	fits_files = [f for f in fits_files if f != ""]
	if len(fits_files) > 3:
		fits_files = fits_files[-3:]
	elif len(fits_files) == 2:
		fits_files = fits_files + [fits_files[1]]
	elif len(fits_files) == 1:
		fits_files = fits_files * 3

	print(fits_files)

	imarrays = [get_image_data(f)[0] for f in fits_files]
	imarrays = [preprocess(im, resize=False) for im in imarrays]
	imarrays = [resize_image(im, size) for im in imarrays]
	opened_files = [fits.PrimaryHDU(data=im) for im in imarrays]
	try:
		dim=pyds9.DS9(FITS_XPA_METHOD)
	except:
		print("Looks like your SAO DS9 is not connected")
		xpa_method = input("Please input you XPA METHOD \n\
				This can be found in your version of ds9:\n\
				\tFile > XPA > Information > XPA METHOD")
		dim = pyds9.DS9(xpa_method)
	outfiles = [fits.HDUList(f) for f in opened_files]

	dim.set("frame new rgb")
	for outf, color in zip(outfiles, ["red", "green", "blue"]):
		dim.set(f"rgb channel {color}")

		dim.set_pyfits(outf)
	if reg_file is not None:
		region_string = utils.get_text_in_file(reg_file)
		dim.set('regions', region_string)
	dim.set("regions -system image")
	dim.set("regions shape box")

	for key, val in config.color_key.items():
		dim.set(f"regions color {val}")
		print(f"Now, annotate all {key}s")
		input("When you are done, press return")

	region_string = dim.get("regions -system image")
	dim.set("frame delete")
	return region_string


def annotate_region_with_size(reg_file, size):
	region_string = utils.get_text_in_file(reg_file)
	lines = region_string.split("\n")
	for i in range(len(lines)):
		line = lines[i]
		if line[:8] == "# size: ":
			region_string = "\n".join(lines[:i] + lines[i + 1:])
	region_string = f"{region_string}\n# size: {size}".strip()
	with open(reg_file, "w") as f:
		f.write(region_string)

def set_region_type_to_image(reg_file, fits_file):
	with fits.open(fits_file) as f:
		region_string = utils.get_text_in_file(reg_file)
		# get the format of the coordinates in the reg file
		region_string_split = region_string.split("\n")
		if len(region_string_split) > 2:
			region_type = region_string_split[2]
		else:
			region_type = None

		if region_type != "image" and region_type != None:
			###
			# hacky code:
			# If you are getting an error here:
			#   FITS_XPA_METHOD needs to be set above
			#   in your version of ds9:
			#       File > XPA > Information > XPA METHOD

			# This is necessary because the pyregions library has bugs
			###
			dim=pyds9.DS9(FITS_XPA_METHOD)

			# get the pixel-image coordinate representation of region 
			outf=fits.HDUList(f[0])
			showtest=dim.set_pyfits(outf)
			dim.set("regions", region_string)
			region_string = dim.get("regions -system image")

			# rewrite region file to be in image coordinates
			with open(reg_file, "w") as reg_file:
				reg_file.write(region_string)

def get_region_from_bbox(row_start, row_end, col_start, col_end, colr):
	row_center = (row_start + row_end) / 2
	col_center = (col_start + col_end) / 2
	row_size = row_end - row_start
	col_size = col_end - col_start
	reg_string = f'# Region file format: DS9 version 4.1\n\
		global color={colr} dashlist=8 3 width=1 \
		font="helvetica 10 normal roman" select=1 highlite=1 dash=0 \
		fixed=0 edit=1 move=1 delete=1 include=1 source=1\n\
		image\n\
		box({col_center},{row_center},{col_size},{row_size}, 0.0)'
	return reg_string



####
# Functions for preprocessing
####

def get_bbox_subimage(imarray, region_mask):
	"""
	IMARRAY:(2d np.array) greyscale image
	REGION_MASK: (2d np.array) 1 at the desired subimage, 0 e/w
	---
	RETURNS: (2D np.array) greyscale image of subregion
	"""
	# get the bbox coordinates
	bbox = get_bbox_from_region_mask(imarray, region_mask)

	# return the area of the image, given the bounding box coordinates
	return subimage_helper(imarray, *bbox)

def get_rect_from_region_mask(imarray, region_mask):
	"""
	IMARRAY:(2d np.array) greyscale image
	REGION_MASK: (2d np.array) 1 at the desired subimage, 0 e/w
	---
	RETURNS: (4-ple) row_start, row_end, col_start, col_end rect coordinates
	"""
	i, j = np.where(region_mask)
	if not len(i) or not len(j):
		print("Region mask is all zero, returning original image")
		return 0, imarray.shape[0] - 1, 0, imarray.shape[1] - 1
	return min(i), max(i), min(j), max(j)


def get_bbox_from_region_mask(imarray, region_mask):
	"""
	IMARRAY:(2d np.array) greyscale image
	REGION_MASK: (2d np.array) 1 at the desired subimage, 0 e/w
	---
	RETURNS: (4-ple) row_start, row_end, col_start, col_end bbox coordinates
	---
	Reference: [2]
	"""

	# get the region of the image where the mask is
	i, j = np.where(region_mask)
	if not len(i) or not len(j):
		print("Region mask is all zero, returning original image")
		return 0, imarray.shape[0] - 1, 0, imarray.shape[1] - 1

	# Get a bounding box on the region
	row_len = max(i) - min(i)
	col_len = max(j) - min(j)

	border_size = imarray.shape[0] // 50

	square_len = max(row_len, col_len) + 2 * border_size

	if square_len >= imarray.shape[0]:
		square_len -= 2 * border_size + 1

	assert square_len < imarray.shape[0], f"bbox must fit in image"

	row_start = min(i) - border_size
	row_end = row_start + square_len
	col_start = min(j) - border_size
	col_end = col_start + square_len

	# make sure bounding box is within image
	if row_start < 0:
		row_start, row_end = 0, row_end - row_start
	elif row_end >= imarray.shape[0]:
		row_start -= (row_end - imarray.shape[0] + 1)
		row_end = imarray.shape[0] - 1
	if col_start < 0:
		col_start, col_end = 0, col_end - col_start
	elif col_end >= imarray.shape[1]:
		col_start -= (col_end - imarray.shape[1] + 1)
		col_end = imarray.shape[1] - 1

	#the bounding box coordinates
	return row_start, row_end, col_start, col_end

def subimage_helper(imarray, row_start, row_end, col_start, col_end):
	"""
	IMARRAY: (2d np.array) greyscale image
	ROW_START: (int) y pos of bottom of bounding box
	ROW_END: (int) y pos of top of bounding box
	COL_START: (int) x pos of left of bounding box
	COL_END: (int) x pos of right of bounding box
	---
	RETURNS: (2d np.array) the subimage defined by the bounding box 
	---
	Reference: [2]
	"""

	# gets a slice to index the array with
	indices = np.meshgrid(np.arange(row_start, row_end), 
						np.arange(col_start, col_end),
						indexing='ij')
	# indexes array with indeces to get image in bounding box
	sub_image = imarray[indices]

	return sub_image

def get_composite_image(images):
	if len(images) > 2:
		return np.dstack(images[-3:])
	elif len(images) == 2:
		return np.dstack(np.concatenate((images, [images[1]])))
	elif len(images) == 1:
		return np.repeat(np.expand_dims(images[0], 2), 3, axis=2)
	else:
		raise "no image given"

def preprocess(imarray, resize=True):
	"""
	IMARRAY: (2d np.array) original image array
	---
	RETURNS: (2d np.array) preprocessed image array
	"""
	imarray = median_filter_nans(imarray)
	imarray = norm_to_256_image(imarray)
	imarray = median_filter_image(imarray, k=5)
	if resize:
		imarray = squeeze_bright(imarray)
		imarray = resize_image(imarray, 128)

	# try to fit it to a gaussian, if it doesn't work, just use empirical 
	# mean and std
	try:
		imarray = truncate_histogram_gaussian(imarray)
	except RuntimeError:
		print("\tWarning: Caught Error: Gaussian Timeout")
		imarray = truncate_histogram(imarray)
	imarray = norm_to_256_image(imarray)
	return imarray

def truncate_histogram(imarray):
	"""
	IMARRAY: (2d np.array) original image array
	---
	RETURNS: (2d np.array) image array clipped at the mean value
	"""
	flat = imarray.flatten()
	std = np.std(flat)
	mean = np.mean(flat)

	threshold = mean

	imarray = imarray.clip(min=threshold)
	return imarray

def truncate_histogram_gaussian(imarray):
	"""
	IMARRAY: (2d np.array) original image array
	---
	RETURNS: (2d np.array)  image array clipped at the mean - 2*std of a
							gaussian that is fit to the image
	---
	REFERENCE: [3]
	"""

	# get the image histogram
	flat = imarray.flatten()
	hist = np.histogram(imarray, bins=200)

	n = len(flat)
	mean = np.mean(flat)
	sigma = np.std(flat)

	# get the x values of the image histogram (avg x of each bin)
	x = (hist[1][1:] + hist[1][:-1])/2
	# get the y values of the image histogram
	y = hist[0]

	# fit a gaussian to the histogram
	popt = scipy.optimize.curve_fit(utils.gaus, x, y, 
									p0=[1,mean,sigma], maxfev=10000)[0]

	# get the mean and standard deviation of said gaussian
	mean_fit, std_fit = popt[1:]

	# clip at the threshold
	threshold = mean_fit - 2 * std_fit
	imarray = imarray.clip(min=threshold)

	return imarray


def resize_image(imarray, new_length):
	"""
	IMARRAY: (2d np.array) original image array
	NEW_LENGTH: (int) size in pixels of width and height of new image
	---
	RETURNS: (2d np.array) new resized interpolated square image array
	"""
	img = norm_to_256_image(imarray)
	img = skimage.transform.resize(img, (new_length, new_length))
	img = norm_to_256_image(img)
	return img

def median_filter_image(imarray, k=9):
	"""
	IMARRAY: (2d np.array) original image array
	K: (int) kernel size for median filter
	---
	RETURNS: (2d np.array) new median filtered image array
	"""
	return scipy.signal.medfilt(imarray, kernel_size=k)

def norm_to_256_image(imarray):
	"""
	IMARRAY: (2d np.array) original image array
	---
	RETURNS: IMARRAY with pixel values normalized between 0 and 255
	"""
	imarray -= np.amin(imarray)
	imarray = np.round((imarray / np.amax(imarray))*255).astype("uint8")
	return imarray

def median_filter_nans(imarray, k = 5):
	"""
	IMARRAY: (2d np.array) original image array
	K: (int) kernel size for median filter
	---
	RETURNS: (2d np.array) new image array, median filtered only on NaNs
	"""
	for i in range(imarray.shape[0]):
		for j in range(imarray.shape[1]):
			# if a pixel is NaN or inf
			if not np.isfinite(imarray[i, j]):
				neighbors = get_neighbors(imarray, i, j, k)
				if neighbors == []:
					# if no non-nan neighbors, set the median to 0
					neighbors = [0]
				# get median
				median = sorted(neighbors)[len(neighbors) // 2]
				# set median where NaN pixel was before
				imarray[i, j] = median
	return imarray

def get_neighbors(imarray, i, j, k):
	"""
	IMARRAY: (2d np.array) original image array
	I: (int) row of pixel to get neighbors of
	J: (int) column of pixel to get neighbors of
	K: (int) size of length of box to find neighbors in
	---
	RETURNS: (list) all finite neighbors of imarray[i, j]
	"""

	# w is the radius of the kernel
	w = (k - 1) // 2

	# out of boundaries areas
	ibound = imarray.shape[0] - 1
	jbound = imarray.shape[1] - 1

	neighbors = []

	# iterate in box around position (i, j) within the boundaries
	for l in range(max(i - w, 0), min(i + w, ibound)):
		for m in range(max(j - w, 0), min(j + w, jbound)):
			# check to only add finite values
			if np.isfinite(imarray[l, m]):
				neighbors.append(imarray[l, m])
	return neighbors


####
# Functions for displaying images
####

def show(imarray, use_log=True, annotations=None):
	"""
	IMARRAY: (2d np.array) original image array
	USE_LOG: (boolean) whether to plot on a logarithmic scale
	ANNOTATIONS: (iterable) zipped boxes and labels
	---
	OUTPUTS: IMARRAY plotted in matplotlib
	"""
	# get rid of NaNs if they exist in the image
	imarray = np.nan_to_num(imarray)
	imarray = imarray.astype(np.uint8)
	fig, ax = plt.subplots(1)
	# plot image
	if use_log:
		norm = LogNorm()
		plt.imshow(imarray, cmap='gray', norm=norm)
	else:
		plt.imshow(imarray, cmap='gray')
	if annotations is not None:
		for box, label in annotations:
			colr = "green" if label == "comet" else "red"
			top, left, bottom, right = box
			rect_args = [(left, top), right - left, bottom - top]
			rect = patches.Rectangle(*rect_args, linewidth=1, 
								edgecolor=colr, facecolor='none')
			ax.add_patch(rect)

	plt.colorbar()
	plt.show(block=False)
	# close image on return press
	input("Press return to Continue")
	plt.close()

def show_fits(fits_file, reg_file = None):
	"""
	FITS_FILE: (str) path to fits image file
	REG_FILE: (str) path to region file or None
	---
	OUTPUTS: plot of fits file with overlaid region file in SAO DS9
	---
	NOTES:  FITS_XPA_METHOD must be properly set at the top of
			the file to open DS9
	"""
	# open ds9
	dim=pyds9.DS9(FITS_XPA_METHOD)
	with fits.open(fits_file) as f:
		outf=fits.HDUList(f[0])
		dim.set("frame new")
		showtest=dim.set_pyfits(outf)
		# set the region file if there is one
		if reg_file:
			region_string = utils.get_text_in_file(reg_file)
			dim.set('regions', region_string)
		dim.set("frame delete")
		# wait until user returns to continue
		input("Press return to Continue")

def show_many(images, row_2 = [], titles=None):
	"""
	IMAGES: (3d np.arrays) array of image arrays
	ROW_2: (3d np.arrays) array of image arrays or None
	TITLES: (list) list of titles
	---
	OUTPUTS: many images plotted in matplotlib on a logarithmic scale
	"""

	# check if a second row of images is desired, set rows and cols
	if len(row_2):
		rows = 2
	else:
		rows = 1
	cols = images.shape[0]

	# create a matplotlib figure to plot images
	fig, axes = plt.subplots(rows, cols, 
		figsize = (int(1.5*len(images)), 3*rows))

	# show images on the figure
	c = 0
	for j in range(cols):
		if len(row_2):
			ax = axes[0][j]
			ax1 = axes[1][j]
			ax1.imshow(row_2[c], cmap="gray")
		else:
			ax = axes[j]
		ax.imshow(images[c], cmap="gray")
		if titles:
			ax.set_title(titles[c])
		c += 1

	# plot and wait for user to return to continue
	plt.show(block=False)
	input("Press return to Continue")
	plt.close()

#incase any other module wants to reference these (say for muting them)
show_functions = ["show", "show_fits", "show_many", "input"]
	

####
# Functions for saving images
####

def save_image(imarray, path):
	"""
	IMARRAY: (2d np.array) image to save
	PATH: (str, ending in .jpg) path to save image
	---
	SAVES TO FILE: (.jpg) grayscale image
	"""
	# show(imarray)
	im = Image.fromarray(imarray, mode="L")
	print(path)
	im.save(path)




##############################################################################
##############################################################################
##############################################################################
#                                  DEPRECATED
##############################################################################
##############################################################################
##############################################################################



def get_normed_data(image_folder, file_to_save=None, file_for_orig=None, 
							mode="comet", reg_files = True):
	"""
	IMAGE_FOLDER: (str) path to original WISE images
	FILE_TO_SAVE: (str) path to put numpy file of preprocessed data or None
	FILE_FOR_ORIG: (str) path to put numpy file of original data or None
	MODE: (str) in ["comet", "one_star", "mul_star", "defect"]
	REG_FILES: (bool) whether or not there are reg files for preprocessing
	---
	RETURNS: (4d np.array)  images zoomed in around the region file and
							preprocessed and sorted by band

							images[2] is an array of images from NEOWISE band 3

							images[1, 9, 0, 3] is pixel [0, 3] of the tenth 
							image on band 2
	---
	SAVES TO FILE:  (.npy) the returned images array
					(.npy) the original images array from the IMAGE_FOLDER
	"""
	# start out with lists (really, they should always be lists, but saving as
	# .npy files is convenient)
	images = [[], [], [], []]
	normal_images = [[], [], [], []]

	# num is used for debugging only, to isolate specific images
	num = 0

	# iterates over the fits and reg files
	for scanfiles in utils.image_iterator(image_folder):
		# iterates over the 4 bands
		for band in range(4):
			fits_file = scanfiles[0][band]
			if reg_files:
				reg_file = scanfiles[1][band]

			print(f"processing {fits_file}")

			# if there is no fits file corresponding to this band, skip this
			if fits_file != "":
				imarray, header = get_image_data(fits_file)

				# keep reference to the original image
				normal_images[band].append(imarray)

				# get all region masks in the image with corresponding mode
				region_masks = []
				if reg_files:
					region_masks = get_region_masks(fits_file, reg_file, mode)

				for reg_mask in region_masks:
					# process image here
					im = get_bbox_subimage(imarray, reg_mask)
					im = preprocess(im)
					images[band].append(im)
		num += 1
	# np.array for easy saving
	images = [np.array(ims) for ims in images]
	images = np.array(images)

	# only save to file if a path is given, otherwise don't worry about it
	if file_to_save:
		np.save(file_to_save, images)
	if file_for_orig:
		np.save(file_for_orig, normal_images)
	return images

def save_training_data(image_folder, save_folder, messier_folder=None):
	"""
	IMAGE_FOLDER: (str) folder of images where the unprocessed training data is
	SAVE_FOLDER: (str) folder to save .jpg images after normalization
	MESSIER_FOLDER: (str) folder to load messier images from, or None
	---
	SAVES TO FILE: in the SAVE_FOLDER, processed .jpg images organized by type
	"""
	# if MESSIER_FOLDER is not given, skip this
	data_f = f"{utils.base_dir()}/data"
	if messier_folder:
		print("\n\n\nGETTING MESSIER DATA\n\n\n")
		messier_data = get_normed_folder_data(messier_folder, 
						orig_file=f"{data_f}/messier/original")
		img_count = 0
		for img in messier_data:
			# create a path if necessary
			path = f"{save_folder}/messier"
			utils.make_folder_if_doesnt_exist(path)
			# save image to path
			save_image(img, f"{path}/messier{img_count}.jpg")
			img_count += 1

	# get all data from the original images + region files
	print("\n\n\nGETTING ONE STAR DATA\n\n\n")
	one_star_data = get_normed_data(image_folder, mode="one_star", 
								file_for_orig=f"{data_f}/one_star/original")
	print("\n\n\nGETTING MUL STAR DATA\n\n\n")
	mul_star_data = get_normed_data(image_folder, mode="mul_star", 
								file_for_orig=f"{data_f}/mul_star/original")
	print("\n\n\nGETTING DEFECT DATA\n\n\n")
	defect_data = get_normed_data(image_folder, mode="defect", 
								file_for_orig=f"{data_f}/defect/original")
	print("\n\n\nGETTING COMET DATA\n\n\n")
	comet_data = get_normed_data(image_folder, mode="comet", 
								file_for_orig=f"{data_f}/comet/original")

	# list for convenient iterating
	data = [(comet_data, "comet"), (one_star_data, "one_star"), 
			(mul_star_data, "mul_star"), (defect_data, "defect")]

	# iterate through each band/category of the data
	for band in range(4):
		for category, cat_name in data:
			img_count = 0
			for img in category[band]:
				path = f"{save_folder}/w{band + 1}/{cat_name}"
				# create a path if necessary
				utils.make_folder_if_doesnt_exist(path)
				# save image to path
				save_image(img, f"{path}/{cat_name}{img_count}.jpg")
				img_count += 1



def get_normed_folder_data(image_folder, save_file=None, orig_file=None):
	"""
	IMAGE_FOLDER: (str) folder of images where the unprocessed training data is
	SAVE_FILE: (str) path to .npy file to save processed images or None
	NORM_FILE: (str) path to .npy file to save original images or None
	---
	RETURNS: (3d np.array) array of preprocessed images
	---
	SAVES TO FILE:  (.npy) the returned images array
					(.npy) the original images array from the IMAGE_FOLDER
	"""
	images = []
	normal_images = []

	# iterate through images in the IMAGE_FOLDER
	for fname in os.listdir(image_folder):
		if fname.endswith("jpg"):
			print(f"processing {fname}")
			path = f"{image_folder}/{fname}"
			# get the grayscale numpy array representation of the image
			im = scipy.ndimage.imread(path, mode="L")

			# save original images
			normal_images.append(im)

			# process the image
			im = preprocess(im)
			images.append(im)

	# np.array for easy saving if desired
	images = np.array(images)

	# only save to file if a path is given, otherwise don't worry about it
	if save_file:
		np.save(save_file, images)
	if orig_file:
		np.save(orig_file, normal_images)
	return images


def squeeze_bright(imarray, k=11):
	"""
	IMARRAY: (2d np.array) original image array
	K: (int)    kernel size of regions whose brightness is considered, 
				i.e. 1 sort of bright pixel will still be ignored unless k == 1
	---
	RETURNS: (2d np.array)  image array tightened around bright pixels
	---
	REFERENCE: [3]
	"""
	# get the brightness of regions through a convolution
	conv = scipy.signal.fftconvolve(imarray, np.ones(shape=(k,k)), mode="same")
	
	# mask where the regions are bright 
	region_mask = conv > (np.mean(imarray) + np.std(imarray)) * k ** 2

	# add some padding
	region_mask[0] = 0
	region_mask[imarray.shape[0] - 1] = 0
	region_mask[:, 0] = 0
	region_mask[:, imarray.shape[1] - 1] = 0

	# return a subimage around the outermost bright pixels
	return get_bbox_subimage(imarray, region_mask)


def rotate_image(imarray, degrees):
	"""
	IMARRAY: (2d np.array) original image array
	DEGREES: (int, multiple of 90) degrees to rotate image by
	---
	RETURNS: IMARRAY rotated DEGREES degrees
	"""
	degrees = (degrees % 360) // 90
	assert degrees in [0, 1, 2, 3], "rotate by multiple of 90 degrees"
	for i in range(degrees):
		imarray = np.rot90(imarray)
	return imarray

def flip_image(imarray, axis="y"):
	"""
	IMARRAY: (2d np.array) original image array
	AXIS: (str, "x" or "y") axis over to flip the image
	---
	RETURNS: IMARRAY flipped along axis MODE
	"""
	if axis == "y":
		imarray = np.fliplr(imarray)
	else:
		imarray = np.flipud(imarray)
	return imarray

def zoom_translate_image(imarray, disturbance_factor = 4):
	"""
	IMARRAY: (2d np.array) original image array
	DISTURBANCE_FACTOR: (int) max pixels to delete from each side of image
	---
	RETURNS: IMARRAY slightly perturbed and then zoomed
	"""
	size = imarray.shape[0]
	left, top = 0, 0
	bottom, right = imarray.shape 

	# randomly choose disturbance and sides to disturb
	disturbance = np.random.randint(1, size//disturbance_factor)
	num_sides = np.random.choice(np.r_[1:5])
	sides = np.random.choice(np.r_[:4], size=num_sides, replace=False)

	# get new bounding box
	for side in sides:
		if side == 0:
			left += disturbance
			top += disturbance
		elif side == 1:
			top += disturbance
			right -= disturbance
		elif side == 2:
			right -= disturbance
			bottom -= disturbance
		else:
			bottom -= disturbance
			left += disturbance
	# get new subimage
	imarray = subimage_helper(imarray, top, bottom, left, right)
	# retain original shape
	imarray = resize_image(imarray, size)
	return imarray


@utils.timeit
def detect_comet_pyramid(imarray, interactive=False, com_dir=None, 
		non_dir=None, space_dir=None, parallelize=True, do_show=False):
	"""
	DOES NOT TAKE PREPROCESSED IMAGES
	IMARRAY: (2d np.array) Full size image that may or may not include a comet
	INTERACTIVE: (bool) If True, ask for feedback and save images
	COM_DIR: (str) if in interactive, place to put comet images
	NON_DIR: (str) if in interactive, place to put non-comet bright images
	SPACE_DIR: (str) if in interactive, place to put empty space images
	PARALLELIZE: (bool) whether or not to use python multiprocessing
	DO_SHOW: (bool) whether to display the image as a matplotlib object
	---
	RETURNS: 	Either the subimage including the comet if a comet in image
				Or the False if a comet is not in the image
	---
	NOTE: 	This function got really ugly with parallelization, visualization!
			If you are trying to read this, ignore all the blocks beginning
			with:
				if parallelize:
			OR
				if do_show:
			That way, the method is only around 15 lines of code
	"""
	# preprocess the entire image at once so we don't have to prepro subims
	processed = preprocess(imarray, resize=False)
	processed = resize_image(imarray, IMSIZE)

	if do_show:
		show(processed, use_log=False)

		def color_func(check):
			if check == True:
				return "green"
			elif check == "maybe":
				return "yellow"
			else:
				return "red"

	if do_show and not interactive:
		use_inter = input("Would you like to switch into interative? \
							interactive allows you to save images back \
							to file[y/N]")
		if use_inter == "y":
			interactive = True

	# parallelization messes with interactivity
	if interactive:
		if com_dir == None:
			com_dir = input("Where to save comet images?")
		if non_dir == None:
			non_dir = input("Where to save bright non-comet images?")
		if space_dir == None:
			space_dir = input("Where to save dark space images?")
		parallelize=False

	if parallelize:
		# get number of cpus, so we can parallelize across all of them
		num_cpu = mp.cpu_count()

		# initialize two queues, one to send images to, and one to recieve
		in_q = mp.Queue(maxsize=num_cpu)
		out_q = mp.Queue()

		# helper function takes a function and returns a function
		# new function adds to a queue instead of returning
		# new func also keeps getting called on elements of queue until queue
		# is killed
		def move_retval_to_q(f, in_q, out_q):
			"""
			F: func to be called as subprocess
			IN_Q: queue to take arguments from
			OUT_Q: queue to place results of function calls on
			---
			NOTES: We also put original arguments on OUT_Q to keep reference
			"""
			def new_func():
				"""
				until we see "Kill" in the queue, run function F on IN_Q 
				elements place initial arguments and output onto OUT_Q
				"""
				while True:
					x = in_q.get()
					if x == "Kill":
						out_q.put(("Process", "Finished"))
						break
					y = f(x[0])
					out_q.put((x, y))
			return new_func

		# parallized version is is_comet that is used by pool workers
		q_is_comet = move_retval_to_q(is_comet, in_q, out_q)
		# runs the while loop in above function, waits for items to be queued
		pool = mp.Pool(num_cpu, initializer=q_is_comet)
		# we close because we do not add any more arguments to the queue
		# each worker only runs q_is_comet, as specified by initializer
		pool.close()

	candidates = []
	zoom_step, orig_zoom = 1.5, 2
	im_mean, im_std = np.mean(processed), np.std(processed)
	if do_show:
		rects = []
	if parallelize:
		curr_ax, curr_ax_num = 0, 0
		p_killed = 0

	# use pyramid proposal to propose comet image candidates
	for subim_info in pyramid_propose(processed, step=64, do_show=do_show,
				interactive=interactive, com_dir=com_dir,non_dir=non_dir, 
				space_dir=space_dir, orig_zoom=orig_zoom, zoom_step=zoom_step):


		# if we need to show the results, we need more info
		if do_show:
			subim, rect_args, ax = subim_info
			if parallelize:
				if curr_ax != ax:
					curr_ax = ax
					curr_ax_num += 1
				draw_data = (rect_args, curr_ax_num)
		else:
			# we set draw_data as None here to pass to in_q which takes tup
			subim, draw_data = subim_info, None

		# get subimage brightness. If the subimage is very dark, we can maybe
		# throw it out
		brightness = np.sum(subim)

		# this should throw out most space images and keep all comet images
		# sufficiently zoomed in
		if brightness < (im_mean + im_std) * subim.size:
			continue

		if parallelize:
			# add image to the queue to be processed
			in_q.put((subim, draw_data))
			while not out_q.empty():
				data, res = out_q.get()
				# confirm a process is killed
				if (data, res) == ("Process", "Finished"):
					p_killed += 1
				else:
					# add candidates to the list
					subim, draw_data = data
					check, score = res
					if check == True:
						candidates.append((subim, score))
					if do_show:
						rect_args, ax_number = draw_data
						missed_zooms = curr_ax_num - ax_number
						zoom_missed = zoom_step ** missed_zooms
						rect_args[0] = (rect_args[0][0] // zoom_missed, 
										rect_args[0][1] // zoom_missed)
						rect_args[1] = rect_args[1] // zoom_missed
						rect_args[2] = rect_args[2] // zoom_missed
						colr = color_func(check)
						rect = patches.Rectangle(*rect_args, linewidth=1,
								alpha=.5, edgecolor="none", facecolor=colr)
						# reset matplotlib patches to not draw over eachother
						for r in rects:
							r.remove()
							r.set_alpha(.03)
						rects = [ax.add_patch(r) for r in rects + [rect]]
						plt.pause(.00001)

		else:
			# check if the subim is a comet
			check, score = is_comet(subim)
			if check == True:
				candidates.append((subim, score))
			if do_show:
				colr = color_func(check)
				rect = patches.Rectangle(*rect_args, linewidth=1, alpha=.5, 
										edgecolor="none", facecolor=colr)
				for r in rects:
					r.set_alpha(.03)
				ax.add_patch(rect)
				rects.append(rect)

	if do_show:
		for r in rects:
			r.remove()
			r.set_alpha(.3 // len(rects))
		fig, ax = plt.subplots(1)
		plt.ion()
		ax.imshow(processed, cmap="gray")
		rects = [ax.add_patch(r) for r in rects]
		plt.pause(.00001)

	if parallelize:
		# indicate that we are finished with the pool by sending "Kill"s
		for _ in range(num_cpu):
			in_q.put("Kill")

		# retrieve images from the out_q, make sure all processes killed
		while p_killed < num_cpu:
			data, res = out_q.get()
			# confirm a process is killed
			if (data, res) == ("Process", "Finished"):
				p_killed += 1
			else:
				check, score = res
				subim, draw_data = data

				# add candidates to the list
				if check == True:
					candidates.append((subim, score))
				if do_show:
					rect_args, ax_number = draw_data
					to_zoom = (zoom_step ** ax_number) / orig_zoom
					rect_args[1] = rect_args[1] * to_zoom
					rect_args[2] = rect_args[2] * to_zoom
					colr = color_func(check)
					rect = patches.Rectangle(*rect_args, linewidth=1,
							alpha=.5, edgecolor="none", facecolor=colr)
					# reset matplotlib patches to not draw over eachother
					for r in rects:
						r.remove()
						r.set_alpha(.03)
					rects = [ax.add_patch(r) for r in rects + [rect]]
					plt.pause(.0001)

		# terminate pool
		pool.join()

	if do_show:
		for r in rects:
			r.set_alpha(.03)
		plt.pause(.01)
		input("Press Return to Continue")
		plt.ioff()
		plt.close()
	print(len(candidates))
	# if there are no comet candidates, there are no comets in image
	if candidates != []:
		# get best comet candidate
		best_im, best_score = max(candidates, key=lambda x:x[1])

		print(f"Best score is {best_score}")
		if do_show:
			show(best_im, use_log=False)
		return True

	return False


def is_comet(imarray):
	"""
	IMARRAY: (2d np.array) 128x128 image that may or may not include a comet
	---
	RETURNS: (bool) whether a comet is in the subimage
	---
	REFERENCE: [4]
	"""
	model_file = "tf_retrainable/tf_files/retrained_graph.pb"
	label_file = "tf_retrainable/tf_files/retrained_labels.txt"
	input_height = 128
	input_width = 128
	input_mean = 128
	input_std = 128
	input_layer = "Mul"
	output_layer = "final_result"

	graph = load_graph(model_file)

	# composite image from single array just repeats
	imarray = get_composite_image([imarray])
	t = read_tensor_from_image_array(imarray)

	input_name = "import/" + input_layer
	output_name = "import/" + output_layer
	input_operation = graph.get_operation_by_name(input_name);
	output_operation = graph.get_operation_by_name(output_name);

	# run image through forward pass to classify
	with tf.Session(graph=graph) as sess:
		start = time.time()
		results = sess.run(output_operation.outputs[0],
						{input_operation.outputs[0]: t})
		end=time.time()
	#format results
	results = np.squeeze(results)

	top_k = results.argsort()[-5:][::-1]
	labels = load_labels(label_file)

	print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
	template = "{} (score={:0.5f})"
	for i in top_k:
		print(template.format(labels[i], results[i]))

	# return whether the image is classified as a comet
	if labels[top_k[0]] == "comet":
		if results[top_k[0]] > .95:
			print("\n\nHigh confidence comet\n\n")
			return True, results[top_k[0]]
		else:
			print("\n\nLow confidence comet\n\n")
			return "maybe", results[top_k[0]]
	print("\n\nNot a comet\n\n")
	return False, results[top_k[0]]

def load_labels(label_file):
	"""
	LABEL_FILE: (str) path to file with labels
	---
	RETURNS: (list) list of labels
	---
	REFERENCE: [4]
	"""
	label = []
	proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
	for l in proto_as_ascii_lines:
		label.append(l.rstrip())
	return label

def pyramid_propose(imarray, step=2, zoom_step=1.5, box_size=128, orig_zoom=2, 
					do_show=True, interactive=False, com_dir=None, non_dir=None, 
					space_dir=None):
	"""
	IMARRAY: (2d np.array) original image array
	STEP: (int) number of pixels to move by
	ZOOM_STEP: (float > 1) factor to zoom by after parsing the entire image
	BOX_SIZE: (int) size of image that is proposed
	ORIG_SCALE: (float) original scale to resize image by
	DO_SHOW: (bool) whether to display images along with region
	INTERACTIVE: (bool) whether to ask for feedback on images
	COM_DIR: (str) optional, directory to put comet images in interactive mode
	NON_DIR: (str) optional, directory to put noncomet images in interactive
	---
	YIELDS: (2d np.array) subimage of next proposed region
			if do_show:
				(matplotlib.axes.Axes) axes object that images are shown on
				(list) 	list of arguments to create a rect matplotlib
						object where the subim is on the ax
	"""
	#nans mess with resizing
	imarray = median_filter_nans(imarray)
	imarray = norm_to_256_image(imarray)
	# resize the image (usually we make it larger here so that the bounding
	# box encompasses fewer pixels)
	imarray = resize_image(imarray, int(imarray.shape[0] * orig_zoom))

	# continue if the bounding box is smaller than the image
	while imarray.shape[0] >= box_size:

		# for interactive plotting
		if do_show:
			fig, ax = plt.subplots(1)
			plt.ion()
			ax.imshow(imarray, cmap='gray')
			num_patches = 0
		# iterate through image by step
		for top in range(0, imarray.shape[1], step):
			if top >= imarray.shape[1] - box_size:
				top = imarray.shape[1] - box_size - 1
			for left in range(0, imarray.shape[0], step):
				if left >= imarray.shape[0] - box_size:
					left = imarray.shape[0] - box_size - 1

				bottom = top + box_size
				right = left + box_size

				subim = subimage_helper(imarray, top, bottom, left, right)
				# if showing ims, display image in interactive plot
				if do_show:
					if num_patches > 0:
						rect.remove()
						num_patches -= 1
					colr = (right/imarray.shape[0], box_size/imarray.shape[0],
							bottom/imarray.shape[1])
					rect_args = [(left, top), right - left, bottom - top]
					rect = patches.Rectangle(*rect_args, linewidth=1, 
										edgecolor=colr, facecolor='none')
					ax.add_patch(rect)
					num_patches += 1
					plt.pause(.005)
					if interactive:
						is_com = input("is this a comet, something else\
										bright, or dark space [c/n/S]")
						s = np.random.choice(list("abcdefgh"), size=8)
						s = "".join(s)
						if is_com == "c":
							save_image(subim, f"{com_dir}/{s}.jpg")
						elif is_com == "n":
							save_image(subim, f"{non_dir}/{s}.jpg")
						else:
							save_image(subim, f"{space_dir}/{s}.jpg")
					yield subim, rect_args, ax
				else:
					yield subim
		#downample image, increasing effective bounding box size
		imarray = resize_image(imarray, int(imarray.shape[0] / zoom_step))
		if do_show:
			plt.ioff()
			plt.close()

