import argparse

from .commands.prep import *
from .commands.infer import *
from .commands.train import *

#-=-=-=-#

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
	pass

#-=-=-=-#

class CLI:
	def __init__(self):
		self.parser = argparse.ArgumentParser(
			description = "AI cli",
			formatter_class = CustomFormatter,
			add_help = False
		)

		self.subparsers = self.parser.add_subparsers(
			dest = "mode",
			required = True,
			help = "Mode to run"
		)

		self.parser.add_argument(
			"-h", "--help",
			action = "help",
			help = "Show this help message and exit."
		)

		#-=-=-=-#
		# register commands
		self.commands = [
			Preprocessing(),
			Training(),
			Inference()
		]

		self.command_map = {}

		for cmd in self.commands:
			cmd.add(self.subparsers)
			self.command_map[cmd.name] = cmd

	# main entry
	def parse_args(self):
		args = self.parser.parse_args()

		cmd = self.command_map.get(args.mode)

		if not cmd:
			self.parser.error(f"Unknown mode: {args.mode}")

		cmd.validate(args, self.parser)

		return args