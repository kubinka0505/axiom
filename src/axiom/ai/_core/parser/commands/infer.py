import os
import argparse
from pathlib import Path

from ...vars import EXTENSIONS_AUDIO, EXTENSIONS_CKPTS

#-=-=-=-#

class Inference:
	name = "infer"

	def __init__(self, formatter = argparse.ArgumentDefaultsHelpFormatter):
		self.formatter = formatter

	# Parser builder
	def add(self, subparsers):
		infer = subparsers.add_parser(
			self.name,
			help = "Run inference on audio files using a trained checkpoint.",
			formatter_class = self.formatter,
			add_help = False
		)

		# Groups
		req = infer.add_argument_group("Required arguments")
		sw = infer.add_argument_group("Switch arguments")

		# Required
		req.add_argument(
			"-i", "--file-input",
			type = str,
			metavar = '"str"',
			required = True,
			help = "Path to audio file or directory of audio files to inference."
		)

		req.add_argument(
			"-m", "--file-checkpoint",
			type = str,
			metavar = '"str"',
			required = True,
			help = "Path to model checkpoint file or directory."
		)

		# Switches
		sw.add_argument(
			"-h", "--help",
			action = "help",
			help = "Show this help message and exit."
		)

		return infer

	# Validation + resolution logic
	def validate(self, args, parser):
		if not args.file_input:
			parser.error("Input path is required")

		if os.path.isdir(args.file_input):

			files = Path(args.file_input).glob("*.*")
			files = [str(f.resolve()) for f in files]

			files = [
				f for f in files
				if f.upper().endswith(EXTENSIONS_AUDIO)
			]

			files = sorted(files, key = os.path.getmtime)

			args.file_input = files[-1] if files else None

		if not args.file_input:
			parser.error("No valid input audio found")

		# ckpt
		if not args.file_checkpoint:
			parser.error("Checkpoint path is required")

		if os.path.isdir(args.file_checkpoint):

			ckpts = Path(args.file_checkpoint).glob("*.*")
			ckpts = [str(c.resolve()) for c in ckpts]

			ckpts = [
				c for c in ckpts
				if c.upper().endswith(EXTENSIONS_CKPTS)
			]

			ckpts = sorted(ckpts, key = os.path.getmtime)

			args.file_checkpoint = ckpts[-1] if ckpts else None

		if not args.file_checkpoint:
			parser.error("No valid checkpoint found")

		return args