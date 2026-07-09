import os
import json
import time
import datetime
import warnings
import traceback
import soundfile as sf
from pathlib import Path

warnings.filterwarnings("ignore", category = ResourceWarning)

from ..audio import process_file

#-=-=-=-#

def _load_existing_mapping(json_path: Path) -> dict:
	"""
	Load a previous _labels.json if present, so reruns merge into it
	instead of overwriting it. Missing/corrupt files just start fresh.
	"""
	if not json_path.exists():
		return {}

	try:
		with open(json_path, "r", encoding = "UTF-8") as f:
			return json.load(f)
	except (json.JSONDecodeError, OSError) as e:
		print(f"[WARNING] Could not read existing {json_path.name} ({e}); starting fresh.")
		return {}

def _write_mapping_atomic(json_path: Path, mapping: dict):
	"""
	Write mapping to a temp file then rename over the target, so a crash
	mid-write can't leave a truncated/corrupt labels file.
	"""
	tmp_path = json_path.with_suffix(json_path.suffix + ".tmp")

	with open(tmp_path, "w", encoding = "UTF-8") as f:
		json.dump(mapping, f, indent = "\t")

	os.replace(tmp_path, json_path)

def main(args):
	src = Path(args.dataset)
	dst = Path(args.destination)
	dst.mkdir(parents = True, exist_ok = True)

	if not src.is_dir():
		print("[ERROR] Input directory does not exist or is not a directory.", file = os.sys.stderr)
		raise SystemExit(1)

	json_path = dst / "_labels.json"

	mapping = _load_existing_mapping(json_path)

	start_time = time.time()

	files = list(src.glob("*.wav")) + list(src.glob("*.flac"))
	files.sort(key = lambda f: f.stat().st_size, reverse = True)

	file_positions = {f: i + 1 for i, f in enumerate(files)}

	# precompute all resampling tasks
	tasks = []
	skipped_no_tasks = []

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
		n_before = len(tasks)

		while target >= args.min_rate:
			tasks.append((file, sr, target))
			target -= args.step

		if len(tasks) == n_before:
			skipped_no_tasks.append(file.name)

	if skipped_no_tasks:
		print(
			f"[WARNING] {len(skipped_no_tasks)} file(s) produced no valid targets "
			f"(sr - step < min_rate): {', '.join(skipped_no_tasks[:10])}"
			+ (" ..." if len(skipped_no_tasks) > 10 else "")
		)

	total_tasks = len(tasks)
	processed_this_run = 0
	errors = 0

	SAVE_EVERY = 1

	try:
		for idx, (file, sr_init, target) in enumerate(tasks, start = 1):
			file_pos = file_positions.get(file, "?")

			if args.labels_only:
				out_name = file.name

				if out_name not in mapping:
					mapping[out_name] = int(sr_init / 2)
					processed_this_run += 1

				continue

			basename = file.stem + f"_{target}"
			basename += f"_{file.suffix.lstrip('.')}"

			out_name = basename + args.extension if args.extension.startswith(".") else basename + "." + args.extension
			out_path = dst / out_name

			if out_path.exists() and not args.force:
				mapping.setdefault(out_name, target)

				print(f"[SKIP] {out_name} already exists." + " " * 10, end = "\r")

				continue

			try:
				success = process_file(file, out_path, target)
			except Exception as e:
				success = False
				errors += 1
				print(f"\n[ERROR] {out_name}: {e}")
				traceback.print_exc()

			if success:
				mapping[out_name] = target
				processed_this_run += 1

				percentage = idx / total_tasks * 100
				progress = f"{idx}/{total_tasks} ({percentage:.2f}%)"

				print(
					f"[<{file_pos}/{len(files)}> {progress}] {out_name} @ {target} Hz" + " " * 10,
					end = "\r"
				)
			else:
				errors += 1
				print(f"[ERROR] {out_name}: process_file returned failure" + " " * 10)

			if processed_this_run and processed_this_run % SAVE_EVERY == 0:
				_write_mapping_atomic(json_path, mapping)

	except KeyboardInterrupt:
		print("\n[INFO] Operation interrupted by user.")

	_write_mapping_atomic(json_path, mapping)

	end_time = str(
		datetime.timedelta(seconds = time.time() - start_time)
	)[2:-3]

	print()
	print(f'[INFO] Wrote JSON label to "{json_path}"')
	print(f"[INFO] {len(mapping)} total labels ({processed_this_run} new/updated in {end_time}")