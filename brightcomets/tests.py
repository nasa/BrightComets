"""
Tests for BrightComets detector

Run tests:
>>> python tests.py
Run a specific test, e.g.:
>>> python tests.py TestImageMethods.test_get_data
Run tests without stopping all the way through
>>> python tests.py --quiet
This may take around 15 minutes

If a particular function is breaking in image_methods.py, it may
be a good idea to run the corresponding test here for debugging
"""

import os
import random
import sys
import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import tensorflow as tf

try:
	from brightcomets import utils, image_methods
except ModuleNotFoundError:
	print("Assuming you are running locally")
	import utils, image_methods
	
BASE_DIR = utils.base_dir()
FITS_FOLDER = f"{BASE_DIR}/WISE_Data"
NOT_COMET_FITS_FOLDER = f"{BASE_DIR}/WISE_Data"
FITS_FILE = f"{FITS_FOLDER}/03821b168/03821b168-w4-int-1b.fits"
REG_FILE = f"{FITS_FOLDER}/03821b168/03821b-w4.reg"
COMET_ORIG_NP_IMAGES = f"{BASE_DIR}/data1/comet/original.npy"
COMET_PROCESSED_NP_IMAGES = f"{BASE_DIR}/data1/comet/preprocessed.npy"
NOT_COMET_NP_IMAGES = f"{BASE_DIR}/data1/not_comet/original.npy"
COMET_JPG_IMG_FOLDER = f"{BASE_DIR}/data2/w4/comet"
NOT_COMET_JPG_IMG_FOLDER = f"{BASE_DIR}/data2/messier"

# IMARRAY = random.sample(list(np.load(COMET_ORIG_NP_IMAGES)[3]), 1)[0]

class TestImageMethods(unittest.TestCase):

	def test_get_data(self):
		imarray, header = image_methods.get_image_data(FITS_FILE)
		image_methods.show(imarray)
		self.assertEqual(imarray[100, 100], 880.8705444335938)
		self.assertEqual(header["RA0"], 304.529473280337)

	def test_region_mask(self):
		mask = image_methods.get_region_masks(FITS_FILE, REG_FILE)[0]
		imarray, header = image_methods.get_image_data(FITS_FILE)
		mask = np.invert(mask)
		imarray = mask * imarray
		image_methods.show(imarray)
		self.assertEqual(imarray[362, 75] == 0)

	def test_bounding_box(self):
		reg_mask = np.zeros_like(IMARRAY)
		reg_mask[0][0] = 1
		subim = image_methods.get_bbox_subimage(IMARRAY, reg_mask)
		self.assertEqual(subim.shape[0], 20)

	def test_image_iterator(self):
		for scanfiles in utils.image_iterator(FITS_FOLDER):
			for band in range(3):
				fits_file = scanfiles[0][band]
				if fits_file == "":
					continue
				reg_file = scanfiles[1][band]
				imarray, header = image_methods.get_image_data(fits_file)
				reg_mask = image_methods.get_region_masks(fits_file, reg_file)

	def test_resize(self):
		image_methods.show(IMARRAY)
		IMARRAY = image_methods.resize_image(IMARRAY, 500)
		image_methods.show(IMARRAY)
		self.assertEqual(IMARRAY.shape[0], 500)

	def test_median_filter(self):
		image_methods.show(IMARRAY)
		IMARRAY = image_methods.median_filter_image(IMARRAY)
		image_methods.show(IMARRAY)

	def test_save_data(self):
		save_file = COMET_PROCESSED_NP_IMAGES[:-4]
		save_file_original = COMET_ORIG_NP_IMAGES[:-4]
		ims = image_methods.get_normed_data(FITS_FOLDER, save_file, 
											save_file_original)
		imarray = ims[0][10]
		image_methods.show(imarray)

		image_methods.get_normed_data(
			NOT_COMET_FITS_FOLDER, 
			file_for_orig = NOT_COMET_NP_IMAGES[:-4], 
			reg_files = False)

		self.assertEqual(imarray, np.load(COMET_PROCESSED_NP_IMAGES)[0][10])

	def test_load_data(self):
		ims = np.load(COMET_PROCESSED_NP_IMAGES)[3]
		origs = np.load(COMET_ORIG_NP_IMAGES)[3]
		rands = np.random.choice(np.r_[:ims.shape[0]], size=10, replace=False)
		images = np.array([ims[i] for i in rands])
		originals = np.array([origs[i] for i in rands])
		titles = [f"Image {i}" for i in rands]
		image_methods.show_many(images, originals, titles)

	def test_preprocess(self):
		fig, ax = plt.subplots(1)
		ax.imshow(IMARRAY, cmap="gray")
		plt.show()
		im = image_methods.median_filter_nans(IMARRAY)
		im = image_methods.preprocess(im, resize=False)
		im = image_methods.resize_image(im, 512)
		image_methods.show(im, use_log=False)

	def test_pyramid_propose(self):
		image_methods.show(IMARRAY)
		for sub_im, rect, ax in image_methods.pyramid_propose(IMARRAY, step=64, 
										zoom_step=1.5, box_size=128, orig_zoom=1, 
										do_show=True, interactive=False):
			time.sleep(.005)

	def test_rotate(self):
		image_methods.show(IMARRAY)
		rotated = image_methods.rotate_image(IMARRAY, 90)
		image_methods.show(rotated)
		self.assertEqual(IMARRAY[0, 0], rotated[-1, 0])

	def test_flip(self):
		image_methods.show(IMARRAY)
		flipped = image_methods.flip_image(IMARRAY, axis="y")
		image_methods.show(flipped)
		self.assertEqual(IMARRAY[0, 0], flipped[0, -1])

	def test_zoom_translate(self):
		ims = [IMARRAY] \
			+ [image_methods.zoom_translate_image(IMARRAY) for _ in range(10)]
		image_methods.show_many(np.array(ims))

	def test_save_training_data(self):
		save = f"{BASE_DIR}/data2"
		comet = f"{BASE_DIR}/WISE_Comets"
		messier = f"{BASE_DIR}/Messier"
		image_methods.save_training_data(comet, save, messier)

	def test_save_image(self):
		img = np.array([[200, 200], [200, 100]])
		image_methods.save_image(img, "test_img.jpg")

	@utils.timeit
	def test_detect_comet(self):
		folders = ["comet", "messier", "not_comet"]
		ans = []
		for folder in folders:
			orig_file = f"{BASE_DIR}/data1/{folder}/original.npy"
			origs = np.load(orig_file)
			#if there are 4 bands, get the 4th band
			if origs.shape[0] == 4:
				origs = origs[3]
			rand = np.random.choice(np.r_[:len(origs)])
			imarray = origs[rand]
			photo_dir = f"{BASE_DIR}/tf_retrainable/tf_files/space_photos"
			answer = image_methods.detect_comet_pyramid(imarray, 
				interactive=False,
				com_dir=f"{photo_dir}/comet",
				non_dir=f"{photo_dir}/not_comet", 
				space_dir=f"{photo_dir}/space", 
				parallelize=True, 
				do_show=True)

			ans.append(answer)
		self.assertEqual(ans[0] is not False, True)
		self.assertEqual(ans[1:], [False] * len(ans[1:]))

	def test_detect_comet_frcnn(self):
		folders = ["comet", "messier", "not_comet"]
		ans = []
		for folder in folders:
			orig_file = f"{BASE_DIR}/data1/{folder}/original.npy"
			origs = np.load(orig_file)
			#if there are 4 bands, get the 4th band
			if origs.shape[0] == 4:
				origs = origs[3]
			rand = np.random.choice(np.r_[:len(origs)])
			imarray = origs[rand]
			answer = image_methods.detect_comet(imarray, do_show=True)

			ans.append(answer)
		self.assertEqual(ans[0] is not False, True)
		self.assertEqual(ans[1:], [False] * len(ans[1:]))

	def test_annotate_image(self):
		reg_string = image_methods.annotate_image(FITS_FILE)
		print(reg_string)

	def test_detect_frcnn_long(self):
		for im, name, labels, boxes in utils.tf_record_iterator(
			f"{BASE_DIR}/frcnn_records/test/annotated.record"):
			image_methods.show(im, use_log=False, annotations=zip(boxes, labels))
			answer = image_methods.detect_comet(im, do_show=True)

	def test_detect_compfrcnn_long(self):
		acc, count = (0,0)
		for im, name, labels, boxes in utils.tf_record_iterator(
			f"{BASE_DIR}/data/composite/c/test.record"):
			print(name)
			image_methods.show(im, use_log=False, annotations=zip(boxes, labels))
			answer = image_methods.detect_comet(im, do_show=True, 
						model="faster_rcnn_inception_v2_coco", im_type="composite")
			acc += answer == ("comet" in labels)
			count += 1
			print(f"accuracy: {acc / count}")

	def test_detect_3frcnn_long(self):
		acc, count = (0,0)
		for im, name, labels, boxes in utils.tf_record_iterator(
						f"{BASE_DIR}/data/band_3/c/test.record"):
			print(name)
			image_methods.show(im, use_log=False, annotations=zip(boxes, labels))
			answer = image_methods.detect_comet(im, do_show=True, 
												im_type="band_3")
			acc += answer == ("comet" in labels)
			count += 1
			print(f"accuracy: {acc / count}")

	def test_untrained_comet_detection(self):
		for images in utils.image_iterator(f"{BASE_DIR}/not_trained_comets"):
			fits = images[0]
			arrs = [image_methods.get_image_data(f)[0] for f in fits]
			preprod = [image_methods.preprocess(arr, resize=False) for arr in arrs]
			resized = [image_methods.resize_image(pre, 512) for pre in preprod]
			comp = image_methods.get_composite_image(resized)
			image_methods.show(comp)
			answer = image_methods.detect_comet(preprod, do_show=True, 
														im_type="composite")
			print(answer)


	def test_pipeline(self):
		x = utils.image_iterator(FITS_FOLDER)
		for _ in range(160):
			scanfiles = next(x)
		print(scanfiles[0])
		ims = scanfiles[0]
		ims = [image_methods.get_image_data(im)[0] for im in ims]
		origs = ims.copy()
		image_methods.show_many(np.array(origs))
		preprod = [image_methods.preprocess(im, resize=False) for im in ims]
		resized = [image_methods.resize_image(im, 512) for im in preprod]
		image_methods.show_many(np.array(origs), row_2=np.array(resized))
		composite = image_methods.get_composite_image(resized)
		image_methods.show(composite, use_log=False)
		image_methods.detect_comet(composite, do_show=True, im_type="composite")



if __name__ == '__main__':
	if "--quiet" in sys.argv:
		shows 	= [f"plt.{f}" for f in dir(plt)] \
				+ [f"image_methods.{f}" for f in image_methods.show_functions]

		for funcname in shows:
			exec(f"{funcname} = lambda *args, **kw: None")

	unittest.main()