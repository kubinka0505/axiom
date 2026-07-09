import os
import argparse

#-=-=-=-#

class Preprocessing:
	name = "prep"

	def __init__(self, formatter = argparse.ArgumentDefaultsHelpFormatter):
		self.formatter = formatter

	# Parser builder
	def add(self, subparsers):
		prep = subparsers.add_parser(
			self.name,
			help = "Dataset preprocessing: resample audio and generate JSON labels.",
			formatter_class = self.formatter,
			add_help = False
		)

		# Groups
		req = prep.add_argument_group("Required arguments")
		opt = prep.add_argument_group("Optional arguments")
		sw = prep.add_argument_group("Switch arguments")

		# Required
		req.add_argument(
			"-i", "--dataset",
			type = str,
			metavar = '"str"',
			required = True,
			help = "Directory containing source audio files."
		)

		# Optional
		opt.add_argument(
			"-o", "--destination",
			type = str,
			metavar = '"str"',
			default = None,
			help = "Directory to write resampled audio and JSON labels."
		)

		opt.add_argument(
			"-k", "--step",
			type = int,
			metavar = "int",
			default = 100,
			help = "Frequency step size to subtract from original sample rate."
		)

		opt.add_argument(
			"-r", "--min-rate",
			type = int,
			metavar = "int",
			default = 4000,
			help = "Minimum allowed sample rate after subtraction."
		)

		opt.add_argument(
			"-e", "--extension",
			type = str,
			metavar = "str",
			default = "WAV",
			help = "Output file extension."
		)

		# -------------------------
		# Switches
		# -------------------------
		sw.add_argument(
			"-l", "--labels-only",
			action = "store_true",
			help = "Only generate labels, no audio preprocessing."
		)

		sw.add_argument(
			"-f", "--force",
			action = "store_true",
			help = "Overwrite existing resampled files."
		)

		sw.add_argument(
			"-h", "--help",
			action = "help",
			help = "Show this help message and exit."
		)

		return prep

	# Validation + normalization
	def validate(self, args, parser):
		if not args.dataset:
			parser.error("Dataset directory is required")

		if not os.path.isdir(args.dataset):
			parser.error("Dataset directory is invalid")

		if args.step <= 0:
			parser.error("Step must be > 0")

		if args.min_rate <= 0:
			parser.error("min-rate must be > 0")

		# normalize extension
		args.extension = "." + args.extension.lower().strip(".")

		# destination logic
		if args.labels_only:
			args.destination = args.dataset

		if not args.destination:
			args.destination = "_".join((args.dataset, "resampled"))

		return args