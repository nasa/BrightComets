import tensorflow as tf
import numpy as np
try:
	from object_detection.utils import dataset_util
except ModuleNotFoundError:
	print("Running a limited version, will not be able to retrain")
import image_methods
import utils
import os
import sys
import shutil
import urllib3
from functools import reduce

def main(args=None):
	all_folders = [
		"WISE_Comets",
		"WISE_Not_Comets"
	]
	basedir = utils.base_dir()
	master_folder = f"{basedir}/master_data"
	all_folders = [f"{basedir}/{f}" for f in all_folders]

	for folder in all_folders:
		folder_type = get_folder_type(folder)

		if folder_type == "flat":
			standardize_flat_folder(folder)
		elif folder_type == "nested":
			standardize_nested_folder(folder)
		if args:
			if args.im_type == "composite" or args.im_type == "band_3":
				clean_up_standardized_folder(folder, add_annotations=False)
			else:
				clean_up_standardized_folder(folder)
		else:
			clean_up_standardized_folder(folder)
	merge_func = lambda x, y: merge_standardized_folders(x, y, master_folder)
	reduce(merge_func, all_folders)

	split_test_train(master_folder, split_ratio=(.8, 0, .2))


	if args is not None:
		im_type = args.im_type
		classes = args.classes
	else:
		print("\ncomposite images combine all bands into 1 image\n\
				all_bands trains on individual images using all data\n\
				band_3 trains on only band 3 images")
		options = set(("all_bands", "composite", "band_3"))
		im_type = input(f"choose from: {options}")
		while im_type not in options:
			print("please enter a valid category, as it is listed")
			im_type = input(f"choose from: {options}")
		poss_labels = ["comet", "one_star", "mul_star", "defect", "planet"]
		labels_indexes = input(f"Which labels would you like to use?\n\
						type indexes (starting from zero) from\n\
						{poss_labels},\ne.g. 03 = \
						{poss_labels[0]}, {poss_labels[3]}\n\
						default: {poss_labels[0]}")
		if labels_indexes == "":
			classes = poss_labels[:1]
		else:
			classes = [poss_labels[int(i)] for i in labels_indexes]
	kw = {"im_type" : im_type, 
			"classes" : classes}

	data_folder = f"data/{im_type}/{''.join([x[0] for x in classes])}"
	for dset in ["train", "hold_out", "test"]:
		print(f"Preparing {dset} data")
		args = [f"{master_folder}/{dset}", f"{data_folder}/{dset}.record"]
		create_full_record(*args, **kw)

	label_string = ""
	for index, label in enumerate(classes):
		label_string += f"item {{\n\tid: {index + 1}\n"
		label_string += f"\tname: '{label}'\n}}\n"
	with open(f"{data_folder}/labels.pbtxt", "w") as f:
		f.write(label_string)




def split_test_train(folder, split_ratio=(.7, .1, .2)):
	for n in ["train", "hold_out", "test"]:
		if os.path.exists(f"{folder}/{n}"):
			shutil.rmtree(f"{folder}/{n}")
	f_lst = [f for f in os.listdir(folder) if os.path.isdir(f"{folder}/{f}")]
	np.random.shuffle(f_lst)

	sr = (np.cumsum(split_ratio) * len(f_lst)).astype("int16")
	assert sr[2] == len(f_lst), "split must sum to 1"
	split = f_lst[:sr[0]], f_lst[sr[0]:sr[1]], f_lst[sr[1]:]

	for f, name in zip(split, ["train", "hold_out", "test"]):

		os.makedirs(f"{folder}/{name}")
		for file in f:
			os.rename(f"{folder}/{file}", f"{folder}/{name}/{file}")


def get_folder_type(folder):
	"""
	Returns either "flat", "nested", or "standard"

	This function might not raise an error if the folders aren't structured
	properly. So, be careful! Really, it should be known that the folder is of
	one of the three main types

	see below for what each type means

	flatFolder:
		name1.jpg
		name2.fits
		...

	standardFolder:
		scanframe:
			scanframe-w{band}(blah).fits 
			OR
			scanframe-w{band}(blah).jpg 
			...
			AND
			scan(frame)-w{band}(blah).reg
			OR
			None
			...
		...

	nestedFolder:
		Object:
			Scan id:
				Frame id:
					fits file (.fits)
					...
				...
			...
			Regions:
				region file (possibly no extension), labeled by scanframeband
				...
		...

	"""
	for file in os.listdir(folder):
		if os.path.isdir(f"{folder}/{file}"):
			for file2 in os.listdir(f"{folder}/{file}"):
				if os.path.isdir(f"{folder}/{file}/{file2}"):
					return "nested"
			return "standard"
		elif file.endswith(".jpg") or file.endswith(".fits"):
			return "flat"
	raise "Looks like a problem with the folder you put in"

def standardize_flat_folder(folder):
	"""
	takes a flattened jpg folder structure:
	Folder:
		name1.jpg
		name2.fits
		...

	and turns it into a partially flattened structure:

	Folder:
		name1:
			name1.jpg
		name2:
			name2.fits
		...
	"""
	for file in os.listdir(folder):
		if file.endswith(".jpg") or file.endswith(".fits"):
			name = file[:file.find(".")]
			os.makedirs(f"{folder}/{name}")
			os.rename(f"{folder}/{file}", f"{folder}/{name}/{file}")


def merge_standardized_folders(folder1, folder2, new_folder):
	"""
	create a new folder, include all data in folder1 U folder2
	returns the name of the new file
	"""
	utils.make_folder_if_doesnt_exist(new_folder)

	if folder1 != new_folder:
		for file in os.listdir(folder1):
			if os.path.isdir(f"{folder1}/{file}"):
				if os.path.exists(f"{new_folder}/{file}"):
					shutil.rmtree(f"{new_folder}/{file}")
				shutil.copytree(f"{folder1}/{file}", f"{new_folder}/{file}")
	if folder2 != new_folder:
		for file in os.listdir(folder2):
			if os.path.isdir(f"{folder2}/{file}"):
				if os.path.exists(f"{new_folder}/{file}"):
					shutil.rmtree(f"{new_folder}/{file}")
				shutil.copytree(f"{folder2}/{file}", f"{new_folder}/{file}")

	return new_folder


def clean_up_standardized_folder(folder, add_annotations=True):
	"""
	add_annotations is whether or not the program should prompt if missing
	annotations
	takes a standardized folder structure:
	Folder:
		scanframe:
			scanframe-w{band}(blah).fits 
			OR
			scanframe-w{band}(blah).jpg 
			...
			AND
			scan(frame)-w{band}(blah).reg
			OR
			None
			...
		...
	
	and cleans up filenames and missing files
	Folder:
		scanframe:
			scanframe-w{band}.fits 
			...
			AND
			scanframe-w{band}.reg
			...
		...
	---
	NOTE: 	do not include w1, w2, w3, or w4 in the folder name. It must only
			be where it is expected, after the scanframe
	"""
	for images in utils.image_iterator(folder):
		im_files = images[0]
		reg_files = images[1]
		comp_reg = images[2][0]
		for band in range(4):
			im = im_files[band]
			if im == "":
				continue
			index = im.find(f"w{band + 1}")
			if index == -1:
				index = im.find(".") - 2
			if im.endswith(".jpg"):
				image_methods.convert_jpg_to_fits(im)
				im = f"{im[:-4]}.fits"
			new_im = f"{im[:index + 2]}.fits"
			im_files[band] = new_im
			os.rename(im, new_im)
			
			reg = reg_files[band]
			new_reg = f"{im[:index + 2]}.reg"
			if reg == "" and add_annotations:
				reg_string = image_methods.annotate_image(new_im)
				with open(new_reg, "w") as f:
					f.write(reg_string)
			else:
				reg_files[band] = new_reg
				if os.path.exists(reg):
					os.rename(reg, new_reg)
			if add_annotations:
				image_methods.set_region_type_to_image(new_reg, new_im)
			size = image_methods.get_image_data(new_im)[0].shape[0]
			if add_annotations:
				image_methods.annotate_region_with_size(new_reg, size)

		size = 512
		if comp_reg == "":
			im_files = [f for f in im_files if f != ""]
			i, j = im_files[0].rfind("/") + 1, im_files[0].rfind("-")
			comp_reg = f"{im_files[0][:i]}comp{im_files[0][i:j]}.reg"
			comp_reg_string = image_methods.annotate_composite(im_files, size=size)
			with open(comp_reg, "w") as f:
				f.write(comp_reg_string)
		image_methods.annotate_region_with_size(comp_reg, size)



def standardize_nested_folder(folder):
	"""
	takes a nested folder structure:
	Folder:
		Object:
			Scan id:
				Frame id:
					fits file (.fits)
					...
				...
			...
			Regions:
				region file (possibly no extension), labeled by scanframeband
				...
		...

	and turns it into a partially flattened structure:

	Folder:
		scanframe:
			scanframe-w{band}.fits 
			...
			AND
			scanframe-w{band}.reg
			...
		...
	"""
	os.system(f"find . -name ‘{folder}/*/.DS_Store’ -type f -delete")
	os.system(f"find {folder}/ -mindepth 2 -type f -exec mv -i '{{}}' \
				{folder}/ ';'")
	for file in os.listdir(folder):
		path = f"{folder}/{file}"
		if os.path.isdir(path):
			shutil.rmtree(path)

	folders = []

	for file in os.listdir(folder):
		if not os.path.isdir(folder + "/" + file) and file != ".DS_Store":
			prefix = ""
			for c in file:
				if c == "-":
					break
				prefix += c
			if file.endswith("fits"):
				folders.append(prefix)

				new_path = f"{folder}/{prefix}"
				if not os.path.exists(new_path):
					os.makedirs(new_path)

				real_file = f"{folder}/{file}"
				new_real_file = f"{new_path}/{file}"

				os.rename(real_file, new_real_file)
			elif file.endswith("xlsx"):
				os.rename(f"{folder}/{file}", f"{base_dir}/{file}")

			elif file.endswith("reg") and file != ".DS_Store":
				real_file = f"{folder}/{file}"
				new_real_file = f"{folder}/{file}.reg"
				os.rename(real_file, new_real_file)

	for file in os.listdir(folder):
		if os.path.isfile(folder + "/" + file) and not file.startswith("."):
			prefix = ""
			for c in file:
				if c == "-":
					break
				prefix += c

			special_folder = None
			for f in folders:
				if prefix in f:
					special_folder = f
			if special_folder == None:
				new_path = f"{folder}/{prefix}"
				if not os.path.exists(new_path):
					os.makedirs(new_path)
				special_folder = prefix


			new_path = f"{folder}/{special_folder}"

			real_file = f"{folder}/{file}"
			new_real_file = f"{new_path}/{file}"

			os.rename(real_file, new_real_file)

@utils.timeit
def create_image_record(fits_data, reg_data, classes=["comet"]):
	"""
	fits_file: (str) path to a fits with objects in it
	reg_file: (str) path to a region file with annotations of fits objects
	composite_ims: (bool) whether or not to use band_stacked images
	---
	RETURNS: (tf.train.Example) TFRecord object for an annotated image
	References: [5]
	"""
	image_format = b"jpeg"
	fits_file = fits_data[-1]
	reg_file = reg_data[-1]

	imarray = image_methods.get_image_data(fits_file)[0]

	reg_masks, labels = image_methods.get_region_masks(fits_file, reg_file,
															mode="labels")
	bboxes = [image_methods.get_rect_from_region_mask(imarray, mask) \
				for mask in reg_masks]

	with open(reg_file) as f:
		lines = f.readlines()
	reg_size = int(lines[-1][8:])

	# we have to deal with a few edge cases here, hence the checks
	if bboxes != []:
		temp = list(zip(*[(b, l) for b, l in zip(bboxes, labels) \
								if l in classes]))
		if temp != []:
			bboxes, labels = temp
		else:
			bboxes, labels = [], []

	ymins, ymaxs, xmins, xmaxs = \
		[[bbox[i] / reg_size for bbox in bboxes] for i in range(4)]

	ims = [image_methods.get_image_data(f)[0] for f in fits_data]
	ims = [image_methods.preprocess(im, resize=False) for im in ims]
	ims = [image_methods.resize_image(im, 512) for im in ims]
	imarray = image_methods.get_composite_image(ims)
	image_tensor = image_methods.read_tensor_from_image_array(imarray,
						input_height=512,
						input_width=512,
						nested=False, 
						normalize=False)

	height, width = 512, 512

	filename = fits_file.encode()
	with tf.Session().as_default():
		encoded_image_data = tf.image.encode_jpeg(image_tensor).eval()

	classes = [(classes).index(l) for l in labels]
	labels = [l.encode() for l in labels]
	tf_example = tf.train.Example(features=tf.train.Features(feature={
		"image/height": dataset_util.int64_feature(height),
		"image/width": dataset_util.int64_feature(width),
		"image/filename": dataset_util.bytes_feature(filename),
		"image/source_id": dataset_util.bytes_feature(filename),
		"image/encoded": dataset_util.bytes_feature(encoded_image_data),
		"image/format": dataset_util.bytes_feature(image_format),
		"image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
		"image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
		"image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
		"image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
		"image/object/class/text": dataset_util.bytes_list_feature(labels),
		"image/object/class/label": dataset_util.int64_list_feature(classes),
	}))
	return tf_example

def create_full_record(input_folder, output_path, im_type="all_bands",
						classes=["comet"]):
	output_folder = output_path[:output_path.rindex("/")]
	utils.make_folder_if_doesnt_exist(output_folder)
	if os.path.exists(output_path):
		os.remove(output_path)

	writer = tf.python_io.TFRecordWriter(output_path)
	count = 0

	for scanfiles in utils.image_iterator(input_folder):
		if im_type != "all_bands":
			if im_type == "composite":
				fits_files = [f for f in scanfiles[0] if f != ""]
			elif im_type == "band_3":
				fits_files = [scanfiles[0][2]]
				if fits_files == [""]:
					continue
			reg_files = scanfiles[2]
			print(count)
			print(fits_files)
			print(reg_files)
			rec = create_image_record(fits_files, reg_files, classes)
			writer.write(rec.SerializeToString())
			count += 1
		else:
			# iterates over the 4 bands
			for band in range(4):
				fits_file = [scanfiles[0][band]]
				reg_file = [scanfiles[1][band]]
				if fits_file != [""]:
					print(count)
					print(fits_file)
					print(reg_file)
					rec = create_image_record(fits_file, reg_file, classes)
					writer.write(rec.SerializeToString())
					count += 1
	writer.close()
	print(count)


def get_objects_from_irsa(folder_objects_dict):

	import os

	inf=open("p2007t2.ics")

	epoch=int(float(inf.readline().rstrip().split()[1]) - 2400000.5)

	inf.readline()
	inf.readline()
	inf.readline()

	for line in inf.readlines():
	    dat=line.rstrip().split()
	    name='t2_'+dat[0]
	    outf=name+'.tbl'
	    ecc=float(dat[2])
	    q=float(dat[3])
	    tperi=float(dat[4])
	    node=float(dat[5])
	    argp=float(dat[6])
	    inc=float(dat[7])

	    curlline='curl -o %s --user wise:orare\{wise\} "https://ceres.ipac.caltech.edu/cgi-bin/MOST/nph-most?catalog=wise_neowiser_yr5&obs_begin=2018+06+01&obs_end=2018+06+20&obj_type=Comet&input_type=manual_input&body_designation=%s&epoch=%5i&eccentricity=%.5f&perih_dist=%.5f&inclination=%.5f&arg_perihelion=%.5f&ascend_node=%.5f&perih_time=%.6f&output_mode=Brief" '%(outf,name,epoch,ecc,q,inc,argp,node,tperi)

	    os.system(curlline)

	# for folder, objects in folder_objects_dict.items():
	# 	# utils.make_folder_if_doesnt_exist(folder)
	# 	for obj in objects:
	# 		http = urllib3.PoolManager()
	# 		r = http.request("POST", "http://irsa.ipac.caltech.edu/applications/wise/",
	# 			fields = {"Enter a position coordinates or the position name" : obj, 
	# 					"Specifies the size of the desired L1b or Atlas image cutout. Omitting this constraint will result in visualization/return of entire L1b/Atlas images.\n\nIn multi-input mode, this takes effect only for input tables without a subsize column." : ""})
	# 		print(r.data)




def randomize_names_in(folders):
	"""
	*deprecated*

	If you want to randomize all the names of files in a directory
	This is originally used for the tensorflow retraining tutorial, where
	the training/test sets are fixed based on filename. 
	"""

	for f in FOLDERS:
		num = len(os.listdir(f))
		rands = np.random.choice(np.r_[:num], size=num, replace=False)
		for file in os.listdir(f):
			if file.endswith(".jpg"):
				num = rands[0]
				rands = rands[1:]
				os.rename(f"{f}/{file}", f"{f}/{num}.jpg")
			if file.endswith(".fits"):
				num = rands[0]
				rands = rands[1:]
				os.rename(f"{f}/{file}", f"{f}/{num}.fits")
			if file.endswith(".reg"):
				num = rands[0]
				rands = rands[1:]
				os.rename(f"{f}/{file}", f"{f}/{num}.reg")

if __name__ == "__main__":
	get_objects_from_irsa({"folder" : ["m81"]})
	#main()