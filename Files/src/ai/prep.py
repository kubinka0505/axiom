import argparse

#-=-=-=-#
# Argparse

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
	pass

def parse_args():
	parser = argparse.ArgumentParser(
		description = "Dataset preprocessing: for each audio in input dir, create resampled mono versions at sr - k, sr - 2 * k, ... and emit JSON mapping.",
		formatter_class = CustomFormatter,
		add_help = False
	)

	#-=-=-=-#

	group_required = parser.add_argument_group("Required arguments")
	group_optional = parser.add_argument_group("Optional arguments")
	group_switch = parser.add_argument_group("Switch arguments")

	#-=-=-=-#

	group_required.add_argument(
		"-i", "--dataset",
		type = str,
		metavar = '"str"',
		help = "Directory containing source audio files."
	)

	#-=-=-=-#

	group_optional.add_argument(
		"-o", "--destination",
		type = str,
		metavar = '"str"',
		help = "Directory to write resampled audio and JSON labels."
	)

	group_optional.add_argument(
		"-k", "--step",
		type = int,
		metavar = "int",
		default = 100,
		help = "Frequency step size to subtract from original sample rate (e.g., 100 produces 44900, 44800, ...)."
	)

	group_optional.add_argument(
		"--min-rate",
		type = int,
		metavar = "int",
		default = 4000,
		help = "Do not go below this Nyquist frequency when subtracting steps."
	)
	group_optional.add_argument(
		"-fmt", "--extension",
		type = str,
		metavar = "str",
		default = "WAV",
		help = "Output file extension. Must be a format SoundFile/PyDub can write."
	)

	#-=-=-=-#

	group_switch.add_argument(
		"-g", "--generate-labels-only",
		action = "store_true",
		help = "Instead of preprocessing dataset, generate labels inside --dataset directory."
	)

	group_switch.add_argument(
		"-f", "--overwrite",
		action = "store_true",
		help = "Overwrite existing resampled files instead of skipping."
	)

	group_switch.add_argument(
		"-h", "--help",
		action = "help",
		help = "Shows this message."
	)

	#-=-=-=-#

	for action in group_required._group_actions:
		action.required = True

	#-=-=-=-#

	args = parser.parse_args()

	if args.generate_labels_only:
		args.destination = args.dataset

	if not args.destination:
		args.destination = "_".join((args.dataset, "resampled"))

	args.extension = "." + args.extension.lower().strip(".")

	return args

args = parse_args()

import os
import json
import soundfile as sf
from pathlib import Path
from pydub import AudioSegment

#-=-=-=-#
# Utils

def resample_file(src: str, dst: str, sr: int, sr_init: int = None) -> str | None:
	"""
	Resample an audio file to a target sample rate and convert it to mono using PyDub.

	Args:
		src (str): Input audio file path.
		dst (str): Output audio file path.
		sr (int): Target sample rate.
		sr_init (int, optional): Initial sample rate. Determined from file if None.

	Returns:
		str | None: Output file path if successful, otherwise None.

	Raises:
		RuntimeError: If audio file cannot be read for sample rate detection.
	"""
	if not sr_init:
		try:
			with sf.SoundFile(src) as f:
				sr_init = f.samplerate
		except RuntimeError as e:
			print(f"[ERROR] Failed to read {src} for sample rate detection: {e}", file = os.sys.stderr)
			return None

	filter_str = f"aresample={sr}:cutoff=1,aresample={sr_init}"
	try:
		audio = AudioSegment.from_file(src)
		os.makedirs(os.path.dirname(dst), exist_ok = True)

		ext = os.path.splitext(dst)[1].lstrip(".")
		if not ext:
			ext = "wav"

		audio.export(
			dst,
			format = ext,
			parameters = [
				"-af", filter_str,
				"-ac", "1",
				"-fflags", "+bitexact",
				"-flags:v", "+bitexact",
				"-flags:a", "+bitexact",
				"-map", "0",
				"-map_metadata", "-1",
			]
		)

		return dst
	except Exception as e:
		print(f"[WARNING] PyDub failed for {src} with filter {filter_str}: {e}", file = os.sys.stderr)

	return None

def main():
	src = Path(args.dataset)
	dst = Path(args.destination)
	dst.mkdir(parents = True, exist_ok = True)

	if not src.is_dir():
		print("[ERROR] Input directory does not exist or is not a directory.", file = os.sys.stderr)
		raise SystemExit(1)

	mapping = {}

	files = list(src.glob("*.wav")) + list(src.glob("*.flac"))
	files.sort(key = lambda f: f.stat().st_size, reverse = True)

	# precompute all resampling tasks
	tasks = []

	for file in files:
		if not file.is_file():
			continue
		try:
			with sf.SoundFile(file) as f:
				sr = f.samplerate
		except Exception as e:
			print(f"[WARNING] Skipping {file.name}: cannot read file - {e}")
			continue

		target = sr - args.step
		while target >= args.min_rate:
			tasks.append((file, sr, target))
			target -= args.step

	total_tasks = len(tasks)

	try:
		for idx, (file, sr_init, target) in enumerate(tasks, start = 1):
			if args.generate_labels_only:
				out_name = file.name
				out_path = src / out_name
				mapping[out_name] = int(sr / 2)
			else:
				basename = file.stem + f"_{target}"
				out_name = basename + args.extension if args.extension.startswith(".") else basename + "." + args.extension
				out_path = dst / out_name

				if out_path.exists() and not args.overwrite:
					prit(f"[WARNING] {out_name} already exists.")
				else:
					success = resample_file(file, out_path, target, sr_init)

					if success:
						mapping[out_name] = target
						progress = f"{idx}/{total_tasks} ({(idx / total_tasks * 100):.1f}%)"
						print(f"[{progress}] {out_name} @ {target} Hz" + " " * 10, end = "\r")
					else:
						print(f"[ERROR] {out_name}")
	except KeyboardInterrupt:
		print("\n[INFO] Operation interrupted by user.")

	# save labels
	json_path = dst / "_labels.json"
	with open(json_path, "w", encoding = "UTF-8") as f:
		json.dump(mapping, f, indent = 4)

	# tab delimit
	with open(json_path, "r", encoding = "UTF-8") as f:
		content = f.read()

	with open(json_path, "w", encoding = "UTF-8") as f:
		f.write(content.strip().replace((" " * 4) + '"', '\t"'))

	print(f"[INFO] Wrote label JSON to {json_path}")
	print(f"[INFO] Total generated files: {len(mapping)}")

if __name__ == "__main__":
	main()