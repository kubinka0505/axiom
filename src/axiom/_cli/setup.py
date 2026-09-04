import os
import argparse
from pathlib import Path

from .._core.setup import *
from .._core.helpers.paths import normalize_path
from .._core.helpers.numbers import to_readable, clamp

#-=-=-=-#
# Setup

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
	pass

def _all_excluded(args, group, keyword: str) -> bool:
	return all(
		getattr(args, a.dest)

		for a in group._group_actions
		if a.dest.startswith("exclude_") and keyword in (a.help or "").lower()
	)

parser = argparse.ArgumentParser(
	description = "Audio files parameters estimator.",
	formatter_class = CustomFormatter,
	add_help = False
)

#-=-=-=-#
# Groups

group_required = parser.add_argument_group("Required arguments")
group_optional = parser.add_argument_group("Optional arguments")

group_processing = parser.add_argument_group("Various processing arguments")
group_processing_sr = parser.add_argument_group("Sample rate estimation arguments")
group_processing_sg = parser.add_argument_group("Spectral gate arguments")
group_processing_switch = parser.add_argument_group("Processing switch arguments (processing)")

group_switch_basic = parser.add_argument_group("Basic switch arguments")

#-=-=-=-#
# Required

group_required.add_argument(
	"-i", "--inputs",
	type = str,
	metavar = '"str"',
	nargs = "*",
	default = [],
	help = "Audio file inputs path.\nSupports directories as inputs."
)

#-=-=-=-#
# Optional

group_optional.add_argument(
	"-o", "--output",
	type = str,
	metavar = '"str"',
	default = None,
	help = "Output file path or existing directory.\nWrites to all inputs if empty.\nDoesn't inherit file structure."
)

group_optional.add_argument(
	"-norm", "--normalize",
	type = float,
	metavar = "float",
	default = None,
	help = f"Optional normalization level for output audio files.\nLeaves as is if none.\nAccepts values from range {NORMALIZE_MIN} to {NORMALIZE_MAX} (dB)."
)

group_optional.add_argument(
	"-m", "--model",
	type = str,
	metavar = '"str"',
	default = None,
	help = "Path to model for estimating sample rate.\nIf none, uses heuristic (subjective) estimation."
)

group_optional.add_argument(
	"-v", "--verbosity",
	type = int,
	metavar = "int",
	default = 0,
	choices = (-1, 0, 1, 2),
	help = "Enables verbose logging.\n* `-1` for quiet\n* `0` for simple prints\n* `1` for beautified output.\n* `2` for advanced output."
)

#-=-=-=-#
# Processing

group_processing.add_argument(
	"-ss", "--start",
	type = str,
	metavar = "int{str}",
	default = 0,
	help = "Analyzed audio start point.\nAccepts both n samples and human readable timestamps."
)

group_processing.add_argument(
	"-to", "--duration",
	type = str,
	metavar = "int{str}",
	default = None,
	help = "Analyzed audio duration.\nAccepts both n samples and human readable timestamps.\nIf none or empty returns file length in samples.\nAlways truncated to {n} samples".format(
		n = "{0} × (10 ^ {1})".format(*MAX_DURATION_DISPLAY.lower().split("e")) if "e" in MAX_DURATION_DISPLAY.lower() else MAX_DURATION
	)
)

group_processing.add_argument(
	"-s", "--skip-each",
	type = int,
	metavar = "int",
	default = 1,
	choices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
	help = "Amounts of each samples to skip during analysis.\nCan speed up processing."
)

group_processing.add_argument(
	"-tl", "--trim-threshold-left",
	type = float,
	metavar = "float",
	default = float("inf"),
	help = "Threshold (dBFS) of left side truncation. Empty samples are always removed if active. Active if not None."
)

group_processing.add_argument(
	"-tr", "--trim-threshold-right",
	type = float,
	metavar = "float",
	default = float("inf"),
	help = "Threshold (dBFS) of right side truncation. Empty samples are always removed if active. Active if not None."
)

#-=-=-=-#
# Sample rate processing

group_processing_sr.add_argument(
	"-fft", "--sr-n-fft",
	type = str,
	metavar = "int",
	default = DEFAULT_VALUE_FFT,
	help = "FFT window size used for heuristic frequency estimation."
)

group_processing_sr.add_argument(
	"-k", "--frequency-step",
	type = str,
	metavar = "int",
	default = -1,
	help = f"Approximate frequency step in sample rate estimation.\nAlways truncated to (sample_rate / {STEP_CLAMP_VALUE}) - 1."
)

#-=-=-=-#
# Spectral gate arguments

group_processing_sg.add_argument(
	"-gc", "--spectral-gate-cutoff",
	type = float,
	metavar = "float",
	default = None,
	help = "Spectral gate cutoff in dB."
)

group_processing_sg.add_argument(
	"-gb", "--spectral-gate-bands",
	type = str,
	metavar = "float | int, ... : ...",
	default = "0, inf",
	help = "Spectral gate bands."
)

#-=-=-=-#
# Switch exlusions

group_processing_switch.add_argument(
	"-nsr", "--exclude-sample-rate",
	action = "store_true",
	help = "Do not estimate sample rate."
)

group_processing_switch.add_argument(
	"-nbd", "--exclude-bit-depth",
	action = "store_true",
	help = "Do not estimate bit depth."
)

group_processing_switch.add_argument(
	"-nc", "--exclude-channels",
	action = "store_true",
	help = "Do not estimate channels amount."
)

group_processing_switch.add_argument(
	"-nbr", "--exclude-bit-rate",
	action = "store_true",
	help = "Do not estimate bitrate."
)

group_processing_switch.add_argument(
	"-np", "--exclude-peak",
	action = "store_true",
	help = "Do not estimate peak."
)

#-=-=-=-#
# Switch basic

group_switch_basic.add_argument(
	"-r", "--recursive",
	action = "store_true",
	help = "Enables recursive interation of provided directory in input."
)

group_switch_basic.add_argument(
	"-f", "--force",
	action = "store_true",
	help = "Forcefully write output."
)

group_switch_basic.add_argument(
	"-ncm", "--exclude-metadata",
	action = "store_true",
	help = "Do not copy file metadata such as tags."
)

group_switch_basic.add_argument(
	"-no", "--no-optimize",
	action = "store_true",
	help = "Do not attempt to optimize output file."
)

group_switch_basic.add_argument(
	"-h", "--help",
	action = "help",
	help = "Shows this message."
)

#---#

for action in parser._actions:
	if action.help != argparse.SUPPRESS:
		action.help = action.help.strip() + "\n"

for action in group_required._group_actions:
	action.required = True

args = parser.parse_args()

#-=-=-=-#
# Settings

if _all_excluded(args, group_processing_switch, "estimate") and not args.output:
	logger.error("Nothing to estimate.")
	raise SystemExit(1)

args._calculate_bitrate = True
if not args.exclude_bit_rate:
	missing = []

	if args.exclude_sample_rate:
		missing.append("sample rate")

	if args.exclude_bit_depth:
		missing.append("bit depth")

	if args.exclude_channels:
		missing.append("channel count")

	if missing:
		if len(missing) > 1:
			last = missing.pop()
			message = f"{', '.join(missing)} and {last}"
		else:
			message = missing[0]

		args._calculate_bitrate = False
		logger.warning(f"Cannot estimate bitrate without estimating {message}".strip())

if args.exclude_bit_rate:
	args._calculate_bitrate = False

if not args.inputs[0]:
	args.inputs = choose_files_dialog()

	if not args.inputs:
		logger.error("No inputs selected.")
		raise SystemExit(1)

args.inputs = [f for f in args.inputs if f]

# normalize all paths
expanded = []

for item in args.inputs:
	item = normalize_path(item, relative = False)
	expanded.append(item)

collected = []

# expand directories
for path in expanded:
	p = Path(path)

	if p.is_dir():
		pattern = "**/*" if args.recursive else "*"
		collected += [f.resolve() for f in p.glob(pattern) if f.is_file()]
	elif p.is_file():
		collected.append(p.resolve())
	else:
		logger.warning(f"Ignored invalid path: {path}")

# Validate
filtered = []

for file in collected:
	ext = file.suffix.lower().lstrip(".")

	if ext in EXTENSIONS_VALID:
		filtered.append(str(file))
	else:
		logger.warning(f"Skipping unsupported file: {file}")

if not filtered:
	logger.error("No valid input files found.")
	raise SystemExit(1)

args.inputs = sorted(set(filtered))

if args.output and len(os.path.splitext(args.output)) < 2 and not os.path.exists(args.output):
	if args.force:
		os.makedirs(args.output, exist_ok = True)
	else:
		logger.error("Output directory doesn't exist. Use --force flag to create it.")
		raise SystemExit(1)

#-=-=-=-#
# Verbosity

if args.verbosity > 1:
	logger.setLevel(logging.DEBUG)
elif args.verbosity > 0:
	logger.setLevel(logging.INFO)
elif args.verbosity < 0:
	logger.setLevel(logging.CRITICAL)

if len(args.inputs) > 1 and args.verbosity > -1 and args.output != "":
	logger.error("Can't write to more than 1 file. Performing analysis only.")

	logger.info(
		'Consider setting {color_arg}--output{reset} to {color_value}""{reset} in order to write to inputs.'.format(
			color_arg = Fore.ORANGE,
			color_value = Fore.BLUE,
			reset = Fore.RESET
		)
	)

if args.normalize is not None:
	args.normalize = float(args.normalize)
	args.normalize = clamp(args.normalize, NORMALIZE_MIN, NORMALIZE_MAX)

#-=-=-=-#
# Readable

if args.sr_n_fft is not None:
	args.sr_n_fft = to_readable(args.sr_n_fft)

if args.frequency_step is not None:
	args.frequency_step = to_readable(args.frequency_step)

if args.spectral_gate_cutoff is not None:
	args.spectral_gate_cutoff = to_readable(args.spectral_gate_cutoff)

if args.trim_threshold_left is not None:
	args.trim_threshold_left = to_readable(args.trim_threshold_left)

if args.trim_threshold_right is not None:
	args.trim_threshold_right = to_readable(args.trim_threshold_right)

#print(args)