import os
import re
import math
import numpy as np
from pathlib import Path
from bisect import bisect_left

from ._setup import *

#-=-=-=-#

def hex2ansi(hexcode: str, fore: bool = True) -> str:
	"""
	Return ANSI escape code for 24-bit color from a hex code string.

	Args:
		hexcode (str): Hex color code, e.g. "fc0", "#ffcc00", "F6A".
		fore (bool): If True, produce a foreground code (38;2); otherwise background (48;2).
	Returns:
		str: ANSI escape sequence for that color.
	"""
	hexcode = hexcode.strip().lstrip("#")

	# expand shorthand like "fc0" -> "ffcc00"
	if len(hexcode) == 3:
		hexcode = "".join(c * 2 for c in hexcode)

	if len(hexcode) != 6:
		raise ValueError('Hex code must be 3 or 6 hex characters (after stripping "#").')

	try:
		r = int(hexcode[0:2], 16)
		g = int(hexcode[2:4], 16)
		b = int(hexcode[4:6], 16)
	except ValueError as e:
		raise ValueError(f'Invalid hex color "{hexcode}"') from e

	prefix = "38" if fore else "48"
	return f"\033[{prefix};2;{r};{g};{b}m"

def choose_files_dialog(initialdir: str = os.getcwd()):
	"""
	Open a files selection dialog for choosing audio files.

	Args:
		initialdir (str): Initial directory to open.

	Returns:
		str: Normalized path to the selected files.

	Raises:
		Exception: If no files are selected or if dialog fails.
	"""
	from tkinter import Tk, filedialog

	root = Tk()
	root.withdraw()

	file_paths = filedialog.askopenfilenames(
		title = "Select audio files",
		initialdir = initialdir,
		filetypes = [
			("Audio files", " ".join(list(EXTENSIONS_DISPLAY.values()))),
			*list(EXTENSIONS_DISPLAY.items()),
			("All files", "*.*"),
		],
		defaultextension = "*.wav"
	)

	if not file_paths:
		file_paths = filedialog.askdirectory(
			title = "Select directory with audio files",
			initialdir = initialdir
		)

	file_paths = [os.path.normpath(file) for file in file_paths]

	return file_paths

def file_size(obj) -> str:
	"""
	Outputs the size of the file at the given object in a human-readable format.
	
	Args:
		obj: The path to the file or bytes.
	
	Returns:
		str: File size as a string with appropriate unit (e.g., "10.5 MB").
	
	Raises:
		FileNotFoundError: If the file does not exist.
		ValueError: If the path is not a file.
	"""
	if isinstance(obj, str):
		if not os.path.exists(obj):
			raise FileNotFoundError(f'The file at path "{obj}" does not exist.')

		if not os.path.isfile(obj):
			raise ValueError(f'The path "{obj}" is not a file.')

		size_bytes = os.path.getsize(obj)
	else:
		size_bytes = obj

	units = ["B", "KB", "MB", "GB", "TB", "PB"]
	index = 0

	while size_bytes >= 1024 and index < len(units) - 1:
		size_bytes /= 1024.0
		index += 1

	if size_bytes <= 1024:
		size_bytes = f"{size_bytes:.2f}"
	else:
		size_bytes = str(int(size_bytes))

	return " ".join((size_bytes, units[index]))

def normalize_path(p: Path) -> str:
	"""
	Normalizes a filesystem path relative to the current working directory.

	Expands environment variables and user home (`~`) before resolving to a relative path.

	Args:
		p (Path): Path object or string representing the file path.

	Returns:
		str: Normalized relative path string.

	Example:
		>>> normalize_path(Path("~/Documents/file.txt"))
		'Documents/file.txt'
	"""
	p = Path(os.path.expandvars(p)).expanduser()
	return os.path.relpath(p, Path.cwd())

def clamp(number: int | float, minimum: int | float, maximum: int | float) -> int | float:
	"""
	Clamps the number to desired range.

	Args:
		number (int | float): The clamped number.
		minimum (int | float): The minimum value to clamp `number` to.
		maximum (int | float): The maximum value to clamp `number` to.

	Returns:
		int | float: The clamped number.
	"""
	return min(max(minimum, number), maximum)

def percentage(percent: float, whole: float, rounding: int = 10) -> float:
	"""
	Calculates the percentage value of a part over a whole.

	Args:
		percent (float): The part value.
		whole (float): The total or whole value.
		rounding (int, optional): Number of decimal places to round the result.

	Returns:
		float: The calculated percentage.

	Example:
		>>> percentage(25, 200)
		12.5
	"""
	return round((percent * 100) / whole, rounding)

def snap_next(value, options: list):
	"""
	Snaps a value to the next closest value in a sorted list of options.

	If the value is greater than all options, the last option is returned.

	Args:
		value (numeric): The target value to be snapped.
		options (list): A list of numeric values to snap to.

	Returns:
		numeric: The closest greater than or equal value in the list, or the maximum if none are greater.

	Example:
		>>> snap_next(200, [128, 192, 256])
		256
	"""
	opts = sorted(options)
	idx = bisect_left(opts, value)

	if idx >= len(opts):
		return opts[-1]

	return opts[idx]

def extend_signal(y: np.ndarray, target_length: int) -> np.ndarray:
	"""
	Extends an ndarray by repeating along the last axis until it reaches at least `target_length`.

	If the repeated signal exceeds the target length, it is trimmed to fit exactly.

	Args:
		y (np.ndarray): Input array, last dimension is time axis.
		target_length (int): Desired minimum length along last axis.

	Returns:
		np.ndarray: Extended array with last axis length exactly `target_length`.

	Example:
		>>> extend_signal(np.array([1, 2, 3]), 7)
		array([1, 2, 3, 1, 2, 3, 1])
		
		>>> extend_signal(np.array([[1,2,3],[4,5,6]]), 7)
		array([[1,2,3,1,2,3,1],
			   [4,5,6,4,5,6,4]])
	"""
	original_len = y.shape[-1]

	if original_len >= target_length:
		# just trim last dimension
		slices = [slice(None)] * (y.ndim - 1) + [slice(target_length)]

		return y[tuple(slices)]

	repeats = (target_length + original_len - 1) // original_len # ceil division

	# tile along last dimension
	extended = np.tile(y, reps = [1]* (y.ndim - 1) + [repeats])

	slices = [slice(None)] * (y.ndim - 1) + [slice(target_length)]

	return extended[tuple(slices)]

def perceptual_difference(sr: int, est_sr: int, min_freq: float = 20.0, max_freq: float = 20000.0):
	"""
	Calculates the perceptual difference between the original and estimated sample rates.

	This function estimates how perceptually different two sample rates are by comparing 
	their Nyquist frequencies on a logarithmic (pitch-like) scale. The result is expressed 
	as a relative percentage difference.

	Args:
		sr (int): Original sample rate (Hz).
		est_sr (int): Estimated sample rate (Hz).
		min_freq (float, optional): Minimum frequency to consider for perceptual comparison.
		max_freq (float, optional): Maximum frequency to consider for perceptual comparison.

	Returns:
		float: Perceptual difference as a percentage. Higher values indicate greater perceptual deviation.

	Example:
		>>> perceptual_difference(44100, 32000)
		13.96
	"""
	est_freq = max(min(est_sr / 2, max_freq), min_freq)
	orig_freq = max(min(sr / 2, max_freq), min_freq)

	# perceptual scale: log10 of freq
	log_est = math.log10(est_freq)
	log_orig = math.log10(orig_freq)

	# relative perceptual diff %
	return 100 * abs(log_orig - log_est) / log_orig

def to_samples(value, sr: int) -> int:
	"""
	Converts a time or sample-based value to number of samples.

	Args:
		value (str | int | float): e.g. "2s", "00:01", "20k", "2e2", 5
		sr (int): Sample rate.

	Returns:
		int: Equivalent number of samples.
	"""
	from pytimeparse.timeparse import timeparse

	unitmap = {"k": 3, "m": 6, "g": 9}

	if isinstance(value, (int, float)):
		return int(value)

	value = str(value).strip().lower()

	# treat as sample count if plain integer string
	if value.isdigit():
		return int(value)

	# scientific notation
	try:
		if "e" in value:
			return int(float(value))
	except ValueError:
		pass

	# match number with unit suffix like 22.01k
	match = re.fullmatch(rf"([0-9]*\.?[0-9]+)([" + "".join(unitmap.keys()) + "}])", value)

	if match:
		number, suffix = match.groups()
		return int(float(number) * (10 ** unitmap[suffix]))

	# time string like "2s", "00:01"
	parsed = timeparse(value)

	if parsed is not None:
		return int(parsed * sr)

	return None