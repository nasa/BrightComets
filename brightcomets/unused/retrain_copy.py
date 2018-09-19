import file_organization
import os
import shutil
import utils
import argparse
import time
import multiprocessing as mp
import subprocess
import re
import train_copy
import eval_copy
import export_inference_graph_copy
try:
	from brightcomets import config
except ModuleNotFoundError:
	import config

def main(args):
	model_ckpt = f"{utils.base_dir()}/models/{args.model}/model.ckpt"
	train_folder = f"{utils.base_dir()}/models/"\
				+ f"{args.model}/{args.im_type}/{classes}/train"
	eval_folder = f"{utils.base_dir()}/models/"\
				+ f"{args.model}/{args.im_type}/{classes}/eval"
	train_record_file = f"{utils.base_dir()}/data/"\
					+ f"{args.im_type}/{classes}/train.record"
	eval_record_file = f"{utils.base_dir()}/data/"\
					+ f"{args.im_type}/{classes}/test.record"
	
	files_missing = not (os.path.exists(train_record_file) \
					and os.path.exists(eval_record_file))
	if args.update_records or files_missing:
		file_organization.main(args=args)

	num_evals = len(list(utils.tf_record_iterator(eval_record_file)))

	label_file = f"{utils.base_dir()}/data/{args.im_type}/"\
				+ f"{classes}/labels.pbtxt"
	label_text = utils.get_text_in_file(label_file)
	num_classes = label_text[label_text.rindex("id: ") + 4]
	config_file = f"{utils.base_dir()}/models/{args.model}/pipeline.config"
	config_text = utils.get_text_in_file(config_file)
	config_text = re.sub(r"num_classes: [0-9]+", 
						f"num_classes: {num_classes}", config_text)
	config_text = re.sub(r'label_map_path: "[0-9a-zA-Z/._]*"', 
					f'label_map_path: "{label_file}"', config_text)
	config_text = re.sub(r'fine_tune_checkpoint: "[0-9a-zA-Z/._]*"',
					f'fine_tune_checkpoint: "{model_ckpt}"', config_text)
	config_text = re.sub(r'num_steps: [0-9]+',
					f"num_steps: {args.train_iterations}", config_text)
	config_text = config_text.replace("step: 0", "step: 1")
	config_text = re.sub(r'num_examples: [0-9]+',
					f"num_examples: {num_evals}", config_text)
	if args.eval_all:
		config_text = re.sub(r'max_evals: [0-9]+',
					f"max_evals: {num_evals}", config_text)
	else:
		config_text = re.sub(r'max_evals: [0-9]+',
					f"max_evals: 10", config_text)

	config_lines = config_text.split("\n")
	train, test = [False]*2
	for i in range(len(config_lines)):
		line = config_lines[i]
		if "train_input_reader" in line:
			train, test = True, False
		elif "eval_input_reader" in line:
			train, test = False, True
		if train:
			if "input_path" in line:
				config_lines[i] = line[:line.find("input_path")]\
								+ f'input_path: "{train_record_file}"'
		elif test:
			if "input_path" in line:
				config_lines[i] = line[:line.find("input_path")]\
								+ f'input_path: "{eval_record_file}"'
	config_text = "\n".join(config_lines)
	config_text = re.sub(r'data_augmentation_options {(^([{}])*{^([{}])*})*^([{}])*}', 
		"data_augmentation_options {\n    random_horizontal_flip {\n    }\n    random_rotation90 {\n    }\n    random_vertical_flip {\n    }\n  }",
		config_text)

	with open(f"{config_file}", mode="w") as f:
		f.write(config_text)

	fine_tuned_folder = f"{utils.base_dir()}/models/{args.model}/{args.im_type}"\
			+ f"/{classes}/tuned"

	if args.retrain:
		if os.path.exists(train_folder):
			shutil.rmtree(train_folder)
		os.makedirs(train_folder)

	if int(args.train_iterations):
		args = f"tensorboard --logdir {eval_folder}/, {train_folder}/"
		args = [x.strip() for x in args.split(" ")]
		tensorboard_p = subprocess.Popen(args)

		train_copy.main(train_folder, config_file)
		# eval_copy.main(config_file, train_folder, eval_folder)
		tensorboard_p.terminate()

		if os.path.exists(fine_tuned_folder):
			shutil.rmtree(fine_tuned_folder)
		ckpts = [f for f in os.listdir(train_folder) if f.startswith("model.")]
		ckpts = [f[f.index("-") + 1:f.rindex(".")] for f in ckpts]
		ckpt = max([int(c) for c in ckpts])

		export_inference_graph_copy.main(config_file, f"{train_folder}/model.ckpt-{ckpt}", fine_tuned_folder)

	print("setup complete")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Set up detector \
			e.g. retrain.py --model faster_rcnn_inception_v2_coco --im_type all_bands \
			--classes comet one_star --train_iterations 2000 --update_records \
			--data_folders WISE_data1 WISE_data2 --retrain --eval_all --reannotate")
	parser.add_argument("--model",
					default="faster_rcnn_inception_v2_coco",
					help="model to use, (options: folders in brightcomets/models) \
					(default: faster_rcnn_inception_v2_coco) (Probably don't change)")
	parser.add_argument("--im_type", default="all_bands",
					help="type of image: (default: all_bands)\
						(options: all_bands, composite, or band_3)")
	parser.add_argument("--classes", metavar = "CLASS", nargs="+", 
					default=[x for x in config.color_key],
					help="classes to use, \
						(options: a subset of: items in your config.py color_key) \
						(default: all keys in config.color_key)")
	parser.add_argument("--train_iterations", default="0",
					help="number of iterations to train:\n\
						(default: 0, i.e. no training)")
	parser.add_argument('--update_records', action='store_true',
					help='reorganizes folders and preprocesses training images\
							use this when you have added new images')
	parser.add_argument('--data_folders', metavar = "FOLDER", nargs="+",
							default=["WISE_data"],
							help="folders to pull images and regions from\
								(default: WISE_data")
	parser.add_argument('--retrain', action='store_true',
				help='start training over for the desired classes')
	parser.add_argument('--eval_all', action='store_true',
				help='evaluate on all test examples')
	parser.add_argument('--reannotate', action='store_true',
				help='prompts reannotation on all images')



	args = parser.parse_args()
	print(args.classes)
	classes = "".join([x[0] for x in args.classes])
	main(args)



