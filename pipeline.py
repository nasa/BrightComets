import image_methods

def detect_comet_band_3(fits_image):
	"""
	fit_image: image to look for comet in
	band: the band number to look for the comet in
	---
	This uses a detector trained on a specific band
	"""
	imarray = image_methods.get_image_data(fits_image)[0]
	imarray = image_methods.preprocess(imarray)
	resp = image_methods.detect_comet_frcnn(imarray, do_show=True, 
						im_type="band_3",
						model="faster_rcnn_inception_v2_coco", 
						classes_string="c")
	if resp == True:
		print("Found a comet")
	return resp


def detect_comet_universal(fits_image):
	"""
	fits_image: image to look for comet in
	---
	RETURNS: whether there is a comet in this image
	---
	This uses a detector trained on all 4 bands
	"""
	image = image_methods.get_image_data(fits_image)[0]
	image = image_methods.preprocess(image)
	return image_methods.detect_comet_frcnn(image)

def detect_comet_composite(all_fits_images):
	"""
	all_band_fits_images: list of fits images, from unique bands
	"""
	images = [image_methods.get_image_data(im)[0] for im in all_fits_images]
	images = np.array([image_methods.preprocess(im) for im in images])

	# get a composite image, default uses the three highest bands
	# if fewer than three bands are provided, make something meaninfgul up
	image = image_methods.get_composite_image(images)

	raise NotImplementedError



