import file_organization
import os
import shutil
import utils
import argparse
import re

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
		os.system(f"python train.py --logtostderr \
				--train_dir={train_folder} \
				--pipeline_config_path={config_file}")

		if os.path.exists(fine_tuned_folder):
			shutil.rmtree(fine_tuned_folder)
		ckpts = [f for f in os.listdir(train_folder) if f.startswith("model.")]
		ckpts = [f[f.index("-") + 1:f.rindex(".")] for f in ckpts]
		ckpt = max([int(c) for c in ckpts])

		os.system(f"python export_inference_graph.py --input_type image_tensor\
				--pipeline_config_path {config_file}\
				--trained_checkpoint_prefix {train_folder}/model.ckpt-{ckpt}\
				--output_directory {fine_tuned_folder}")
		# os.system(f"python eval.py \
  #   				--logtostderr \
  #   				--pipeline_config_path={config_file}\
  #   				--checkpoint_dir={train_folder}/\
  #   				--eval_dir={eval_folder}/")
	print("setup complete")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Set up detector")
	parser.add_argument("--model",
					default="faster_rcnn_inception_v2_coco",
					help="model to use")
	parser.add_argument("--im_type", default="all_bands",
					help="type of image, all_bands, composite, or band_3")
	parser.add_argument("--classes", metavar = "CLASS", nargs="+", 
					default=["comet"],
					help="classes to use, a subset of:\n\
						comet one_star mul_star defect planet")
	parser.add_argument("--train_iterations", default="0",
					help="number of iterations to train:\n\
						default is 0, no training")
	parser.add_argument('--update_records', action='store_true',
					help='reorganizes folders and preprocesses training images\
							use this when you have added new images')
	parser.add_argument('--retrain', action='store_true',
				help='start training over')




	args = parser.parse_args()
	print(args.classes)
	classes = "".join([x[0] for x in args.classes])
	main(args)



