import tensorflow as tf
import numpy as np
try:
	from object_detection.utils import dataset_util
except ModuleNotFoundError:
	print("Running a limited version, will not be able to retrain")
import os
import sys
import shutil
import urllib3
from functools import reduce

try:
	from brightcomets import image_methods, utils, config
except ModuleNotFoundError:
	print("Assuming you are running locally")
	import image_methods, utils, config

IMSIZE = config.im_size
def main(args=None):
	"""
	called by the retrain.py script
	args is the command line args to retrain.py

	reorganizes files and creates TFRecord files for images and regions
	"""
	if args == None:
		assert "this function should be called with command line arguments (from retrain.py)"
	all_folders = args.data_folders
	im_type = args.im_type
	classes = args.classes
	reannotate = args.reannotate

	basedir = utils.base_dir()
	master_folder = f"{basedir}/master_data"
	utils.make_folder_if_doesnt_exist(master_folder)
	all_folders = [f"{basedir}/{f}" for f in all_folders]


	assert len(all_folders) > 0, "must specify data folder"
	for f in all_folders:
		assert os.path.exists(f), "path to data folder must exist"

	for folder in all_folders:
		standardize_folder(folder)
		set_all_to_image_coords(folder)
		add_annotations(folder, im_type=im_type, reannotate=reannotate)
		
	
	if len(all_folders) > 1:
		merge_func = lambda x, y: merge_standardized_folders(x, y, master_folder)
		reduce(merge_func, all_folders)
	elif len(all_folders) == 1:
		for f in os.listdir(all_folders[0]):
			shutil.copytree(f"{all_folders[0]}/{f}", f"{master_folder}/{f}")

	split_test_train(master_folder, split_ratio=(.8, 0, .2))
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

def add_annotations(folder, im_type="all_bands", reannotate=False):
	if im_type == "all_bands":
		iterator = ((x[0][i], x[1][i]) for i in range(4) \
					for x in utils.image_iterator(folder))
	else:
		iterator = utils.image_iterator(folder)
	print("\n\n\n\n\nAdding annotations\n\n\n\n\n\n\n\n\n\n\n")
	for scanfiles in iterator:
		if im_type == "composite":
			fits, reg = scanfiles[0], scanfiles[2][0]
			if any(fits) and (reg == "" or reannotate):
				fname = str([x for x in fits if x != ""][0])
				reg_name = f"{fname[:fname.find('-')]}-comp.reg"
				size = IMSIZE
				if reg == "":
					reg_string = image_methods.annotate_composite(fits)
				else:
					reg_string = image_methods.annotate_composite(fits, reg)
			else:
				reg_string = utils.get_text_in_file(reg)
		else:
			if im_type == "band_3":
				fits, reg = scanfiles[0][2], scanfiles[1][2]
			else:
				fits, reg = scanfiles
			if (reg == "" or reannotate) and fits != "":
				if reg == "":
					reg_string = image_methods.annotate_image(fits)
				else:
					reg_string = image_methods.annotate_image(fits, reg)
			else:
				reg_string = utils.get_text_in_file(reg)
			reg_name = f"{fits[:-5]}.reg"
			size = image_methods.get_image_data(fits)[0].shape[0]
		with open(reg_name, "w") as f:
			f.write(reg_string)
		print("GOING TO REGION")
		image_methods.annotate_region_with_size(reg_name, size)


def standardize_folder(folder):
	base = utils.base_dir()
	regions = f"{base}/regions"
	static_regions = f"{base}/regions-static"
	fits = f"{base}/fits"

	copy_to_folder_by_type(static_regions, regions, ".reg")
	copy_to_folder_by_type(folder, regions, ".reg", overwrite=False)
	deep_jpg_to_fits(folder)
	copy_to_folder_by_type(folder, fits, ".fits")
	trim_name_fat(regions)
	trim_name_fat(fits)
	clean_folder = f"{folder}-clean"
	copy_to_folder_by_type(regions, clean_folder, ".reg")
	copy_to_folder_by_type(fits, clean_folder, ".fits", overwrite=False)
	# clean folder is flattened, as it must be
	group_by_prefix(clean_folder)
	remove_unnecessary_folders(clean_folder)
	shutil.rmtree(folder)
	shutil.move(clean_folder, folder)

def set_all_to_image_coords(folder):
	for scanfiles in utils.image_iterator(folder):
		fits = scanfiles[0]
		regions = scanfiles[1]
		for band in range(4):
			if fits[band] and regions[band]:
				image_methods.set_region_type_to_image(regions[band], fits[band])

def remove_unnecessary_folders(folder):
	for f in os.listdir(folder):
		if os.path.isdir(f"{folder}/{f}"):
			files = os.listdir(f"{folder}/{f}")
			if not any([x.endswith(".fits") for x in files]):
				shutil.rmtree(f"{folder}/{f}")


def group_by_prefix(flat_folder):
	pref = lambda f: f[:f.find("-")] if f.find("-") != -1 else f[:f.rindex(".")]
	prefp = lambda f: f"{flat_folder}/{pref(f)}"
	[os.makedirs(p) for p in set(prefp(f) for f in os.listdir(flat_folder)) \
	if not os.path.exists(p)]
	for file in [f for f in os.listdir(flat_folder) if \
				os.path.isfile(f"{flat_folder}/{f}")]:
		os.rename(f"{flat_folder}/{file}", f"{prefp(file)}/{file}")

def trim_name_fat(folder):
	for folder, file in utils.deep_listdir(folder):
		file_path = f"{folder}/{file}"
		index = file.find("-", file.find("-") + 1)
		if index != -1:
			file = f"{file[:index]}{file[file.find('.'):]}"
		if file.startswith("comp") and file.endswith(".reg"):
			print(file)
			file = f"{file[4:-4]}-comp.reg"
		print(file)
		os.rename(file_path, f"{folder}/{file}")

def deep_jpg_to_fits(folder):
	for folder, file in utils.deep_listdir(folder):
		if file.endswith(".jpg"):
			image_methods.convert_jpg_to_fits(f"{folder}/{file}")

def copy_to_folder_by_type(source_folder, dest_folder, extension,
								 overwrite=True):
	"""
	move all files ending in extension in source_folder to dest_folder
	"""
	if overwrite:
		utils.make_folder_if_doesnt_exist(dest_folder)
		shutil.rmtree(dest_folder)
	utils.make_folder_if_doesnt_exist(dest_folder)
	for folder, file in utils.deep_listdir(source_folder):
		if file.endswith(extension):
			shutil.copyfile(f"{folder}/{file}", f"{dest_folder}/{file}")



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
	ims = [image_methods.resize_image(im, IMSIZE) for im in ims]
	imarray = image_methods.get_composite_image(ims)
	image_tensor = image_methods.read_tensor_from_image_array(imarray,
						input_height=IMSIZE,
						input_width=IMSIZE,
						nested=False, 
						normalize=False)

	height, width = IMSIZE, IMSIZE

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
			reg_files = [scanfiles[1][2]]
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

def fix_object_detection_compatibility():
	path1 = f"{utils.base_dir()}/../models/research/object_detection/model_lib.py"
	path2 = f"{utils.base_dir()}/../models/research/object_detection/metrics/coco_tools.py"
	path3 = f"{utils.base_dir()}/../models/research/object_detection/utils/learning_schedules.py"
	path4 = f"{utils.base_dir()}/../models/research/object_detection/utils/object_detection_evaluation.py"

	with open(path1, "r") as f:
		lines = f.readlines()
		lines[281] = "      [loss_tensor for loss_tensor in losses_dict.values()]\n"
		lines[390] = "      eval_metric_ops = {str(k): v for k, v in eval_metric_ops.items()}\n"
		text = "".join(lines)
	with open(path1, "w") as f:
		f.write(text)
	with open(path2, "r") as f:
		lines = f.readlines()
		lines[117] = "    results.dataset['categories'] = copy.deepcopy(list(self.dataset['categories']))\n"
		text = "".join(lines)
	with open(path2, "w") as f:
		f.write(text)
	with open(path3, "r") as f:
		lines = f.readlines()
		lines[167] = "                                      list(range(num_boundaries)),\n"
		text = "".join(lines)
	with open(path3, "w") as f:
		f.write(text)
	with open(path4, "r") as f:
		lines = f.readlines()
		lines[841] = "      print('Scores and tpfp per class label: {0}'.format(class_index))\n"
		lines[842] = "      print(tp_fp_labels)\n"
		lines[843] = "      print(scores)\n"
		text = "".join(lines)
	with open(path4, "w") as f:
		f.write(text)

def write_to_bash_profile():
	bash_profile = f"{os.path.expanduser('~')}/.bash_profile"
	pwd = f"{utils.base_dir()}/../models/research"
	text = f"\n# From BrightComets/models/research/\nexport PYTHONPATH=$PYTHONPATH:{pwd}:{pwd}/slim"
	curr_text = utils.get_text_in_file(bash_profile)
	if text not in curr_text:
		with open(bash_profile, "a") as f:
			f.write(text)







##############################################################################
##############################################################################
##############################################################################
#                                  DEPRECATED
##############################################################################
##############################################################################
##############################################################################



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
		size = 512
		if comp_reg == "":
			im_files = [f for f in im_files if f != ""]
			j = im_files[0].rfind("-")
			comp_reg = f"{im_files[0][:j]}-comp.reg"
			comp_reg_string = image_methods.annotate_composite(im_files,
															 size=size)
			with open(comp_reg, "w") as f:
				f.write(comp_reg_string)

		image_methods.annotate_region_with_size(comp_reg, size)
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
				if comp_reg != "":
					print("Composite region for reference:")
					print(utils.get_text_in_file(comp_reg))
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

			elif not file.endswith("reg") and file != ".DS_Store":
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
	fix_object_detection_compatibility()
	write_to_bash_profile()