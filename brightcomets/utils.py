import time
import os
import sys

import numpy as np
import scipy
import arrr
import datetime
import tensorflow as tf

def timeit(method):
	"""
	METHOD: function to decorate
	---
	RETURNS: decorated function
	"""
	def timed(*args, **kw):
		"""
		ARGS: arguments to path to METHOD
		KW: key word arguments to path to METHOD
		RETURNS: return value of running METHOD(*ARGS, **KW)
		---
		OUTPUTS: time that the function took to run in seconds
		"""
		ts = time.time()
		result = method(*args, **kw)
		te = time.time()
		print(f"{method.__name__}: {(te-ts):.2f} s")
		return result
	return timed

def path_exists(relative_path):
	"""
	RELATIVE_PATH: relative path to a file in the current directory
	---
	RETURNS: Whether that path exists
	"""
	# create an absolute path to the same file
	base_dir = os.path.dirname(os.path.realpath(__file__))
	path = base_dir + (relative_path[0] != "/") * "/" + relative_path
	return os.path.exists(path)



def base_dir():
	"""
	RETURNS: the path to the current (BrightComets) directory
	"""
	return os.path.dirname(os.path.realpath(__file__))

def make_folder_if_doesnt_exist(name):
	"""
	NAME: folder name
	---
	SAVES TO FILE: the folder named NAME, iff it doesn't already exist
	"""
	if not path_exists(name) and not os.path.exists(name):
		os.makedirs(name)

def get_text_in_file(file):
	"""
	FILE: path to file to read text from
	---
	RETURNS: text in file
	"""
	with open(file, 'r') as f:
		return f.read()

def image_iterator(img_folder):
	"""
	IMG_FOLDER:	(str) path to folder of infrared images. 
				must be organized such that:
				img_folder contains only subfolders
				each subfolder has regions and observations for one scan/fram
				each subfolder has 4 IR bands of regions/observations
				This is consistent with the WISE_Comets folder, see README

	YIELDS: a 2d array, images[0][:] = fits files, images[1][:] = reg files
			images[:][3] gives fits and reg files for band 3
			images contains images for a single scan/frame
			images[2] is a region file for the composite image
	"""

	# iterate through folders, each folder has images
	for sub_folder in os.listdir(img_folder):
		images = [["", "", "", ""], ["", "", "", ""], [""]]
		sub_folder_path = f"{img_folder}/{sub_folder}"
		if sub_folder != ".DS_Store":
			for file in os.listdir(sub_folder_path):
				path = f"{sub_folder_path}/{file}"
				file_type = (file.endswith("reg") or file.find(".") == -1)\
						 + ("comp" in file)
				band_number = 0
				for i in range(len(file) - 1):
					indicator = file[i]
					number = file[i + 1]
					if indicator == "w":
						band_number = int(number) - 1
				# path gets put in the right place with file type and band num
				images[file_type][band_number] = path
			yield images

def deep_listdir(folder):
	for file in os.listdir(folder):
		file_path = f"{folder}/{file}"
		if os.path.isdir(file_path):
			yield from deep_listdir(file_path)
		else:
			yield folder, file

def tf_record_iterator(tf_record_file):
	"""
	REFERENCE: [6]
	"""
	record_iterator = tf.python_io.tf_record_iterator(path=tf_record_file)

	for string_record in record_iterator:
		example = tf.train.Example()
		example.ParseFromString(string_record)
		img_string = example.features.feature['image/encoded'].bytes_list.value[0]
		name = example.features.feature["image/filename"].bytes_list.value[0]
		name = name.decode()
		labels = example.features.feature['image/object/class/text'].bytes_list.value
		bboxs = [example.features.feature[f"image/object/bbox/{coord}"].float_list.value \
				for coord in ["ymin", "xmin", "ymax", "xmax"]]
		bboxs = np.dstack(bboxs)[0]
		labels = [l.decode() for l in labels]
		img_array = tf.image.decode_jpeg(img_string)
		with tf.Session().as_default():
			img_array = img_array.eval()

		bboxs[:, 0] *= img_array.shape[0]
		bboxs[:, 1] *= img_array.shape[1]
		bboxs[:, 2] *= img_array.shape[0]
		bboxs[:, 3] *= img_array.shape[1]

		yield img_array, name, labels, bboxs

def gaus(x,a,x0,sigma):
	"""
	X: (float) actual value
	A: (float) scale
	X0: (float) mean
	SIGMA: (float)standard deviaton
	---
	RETURNS: scaled probability of outputing X from a gaussian random variable
	"""
	return a*scipy.exp(-(x-x0)**2/(2*sigma**2))

def special_print(verbose=True):
	def print_helper(*args, **kw):
		for text in args:
			if not isinstance(text, str):
				text = text.__repr__()
			if verbose:
				now = datetime.datetime.now()
				if "pirate" in sys.argv or (now.month==9 and now.day==19):
					text = arrr.translate(text)
		print(*args, **kw)
	return print_helper

