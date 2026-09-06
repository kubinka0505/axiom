import time
import logging
from datetime import timedelta

from .._core.main import Axiom

from .helper import process_file, clear_colored_text_file
from .setup import *

def main():
	__START = time.time()

	try:
		axiom = Axiom(args.file_input, recursive = args.recursive)

		for idx, file in enumerate(axiom.files):
			process_file(
				idx,
				file,
				len(axiom.files),
				args
			)

		logger.info(
			"Processed {n_files} file{suffix} in at most {time}.".format(
				n_files = len(axiom.files),
				suffix = "s" if len(axiom.files) > 1 else "",
				time = str(timedelta(seconds = time.time() - __START))[2:-3],
			)
		)

		# clear all logger file handlers
		for handler in logger.handlers:
			if isinstance(handler, logging.FileHandler):
				file_path = handler.baseFilename

				if os.path.exists(file_path):
					clear_colored_text_file(file_path)
	except KeyboardInterrupt:
		msg = "Operation interrupted by user."

		if args.verbosity > 0:
			logger.info(msg)
		else:
			print(msg)