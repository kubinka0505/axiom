import os
import argparse
from pathlib import Path

#-=-=-=-#

class Training:
	name = "train"

	def __init__(self, formatter = argparse.ArgumentDefaultsHelpFormatter):
		self.formatter = formatter

	# Parser builder
	def add(self, subparsers):
		train = subparsers.add_parser(
			self.name,
			help = "Train model on cutoff frequency regression task.",
			formatter_class = self.formatter,
			add_help = False
		)

		# Groups
		req = train.add_argument_group("Required arguments")
		opt = train.add_argument_group("Optional arguments")
		sw = train.add_argument_group("Switch arguments")

		# Required
		req.add_argument(
			"-i", "--file-input",
			type = str,
			metavar = '"str"',
			help = "Checkpoint to resume training from."
		)

		# Required
		req.add_argument(
			"-d", "--dataset",
			type = str,
			metavar = '"str"',
			required = True,
			help = "Directory with label and audio files."
		)

		req.add_argument(
			"-e", "--epochs",
			type = int,
			metavar = "int",
			required = True,
			help = "Number of epochs to train."
		)

		# Optional
		opt.add_argument(
			"-l", "--labels",
			type = str,
			metavar = '"str"',
			default = None,
			help = "Path to labels mapping file. If omitted, auto-detect first .json in dataset."
		)

		opt.add_argument(
			"-o", "--output-directory",
			type = str,
			metavar = '"str"',
			default = os.path.join("logs", "models"),
			help = "Checkpoint output directory."
		)

		opt.add_argument(
			"-bs", "--batch-size",
			type = int,
			metavar = "int",
			default = 8,
			help = "Batch size for training/validation."
		)

		opt.add_argument(
			"-lr", "--learning-rate",
			type = float,
			metavar = "float",
			default = "2e-4",
			help = "Learning rate for optimizer."
		)

		opt.add_argument(
			"-see", "--save-each-epoch",
			type = int,
			metavar = "int",
			default=10,
			help = "Save checkpoint every N epochs."
		)

		opt.add_argument(
			"-eee", "--evaluate-each-epoch",
			type = int,
			metavar = "int",
			default = 1,
			help = "Evaluate checkpoint every N epochs."
		)

		opt.add_argument(
			"-p", "--patience",
			type= int,
			metavar ="int",
			default = 10,
			help = "Early stopping patience (no improvement)."
		)

		opt.add_argument(
			"-logs", "--logs-directory",
			type = str,
			metavar = '"str"',
			default = "logs",
			help = "TensorBoard logs and model output directory."
		)

		# Switches
		sw.add_argument(
			"-ntb", "--no-tensorboard",
			action = "store_true",
			help = "Disable TensorBoard logging."
		)

		sw.add_argument(
			"-h", "--help",
			action = "help",
			help = "Show this help message and exit."
		)

		return train

	# Validation
	def validate(self, args, parser):
		if args.mode.lower() == "infer":
			if not args.file_input:
				parser.error("Input is required for inference mode")
			elif not os.path.exists(args.file_input):
				parser.error(f"Input not found: {args.file_input}")
			elif not os.path.isfile(args.file_input):
				parser.error(f"Input must be a file, not a directory: {args.file_input}")

		else: # train
			if not args.dataset or not os.path.exists(args.dataset):
				parser.error("Valid dataset directory is required for training")

			if args.file_input:
				if not os.path.exists(args.file_input):
					parser.error(f"Resume checkpoint not found")
				if not os.path.isfile(args.file_input):
					parser.error(f"Resuming must have a checkpoint file, not a directory")

		if args.epochs is None or args.epochs <= 0:
			parser.error("Epochs must be > 0")

		if args.batch_size <= 0:
			parser.error("Batch size must be > 0")

		if args.save_each_epoch <= 0:
			parser.error("Save-each-epoch must be > 0")

		if args.patience <= 0:
			parser.error("Patience must be > 0")

		# learning rate safety
		try:
			args.learning_rate = float(args.learning_rate)
		except Exception:
			parser.error("Could not parse learning rate")

		# labels auto-discovery
		if not args.labels:
			label_files = list(Path(args.dataset).glob("*.json"))

			if label_files:
				args.labels = str(label_files[0])

		if not args.labels:
			parser.error("No labels file found! (provide --labels)")

		# sanity adjustments
		if args.save_each_epoch > args.epochs:
			args.save_each_epoch = args.epochs

		return args