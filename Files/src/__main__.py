import warnings
import soundfile as sf
from pathlib import Path

from typing import Any, List, Dict

from ._setup import *
from ._helper import *
from ._main import Estimators as Estimators

#-=-=-=-#

class Axiom:
	def __init__(self, filepaths: List[str | Path], recursive: bool = False) -> None:
		"""
		Initialize with a list of file paths or directories, optionally recursively.

		Args:
			filepaths (list[str | Path]): List of file or directory paths.
			recursive (bool): Whether to recursively search directories.

		Raises:
			ValueError: If no valid audio files are found or paths are invalid.
		"""
		if isinstance(filepaths, (str, Path)):
			filepaths = [filepaths]

		self.files = []

		for path in filepaths:
			path = Path(path)

			if path.is_file():
				self.files.append(normalize_path(path))

			elif path.is_dir():
				glob_pattern = "**/*" if recursive else "*"

				for file in path.glob(glob_pattern):
					if file.is_file():
						self.files.append(normalize_path(file))

			else:
				raise ValueError(f"Invalid path: {path}")

		# unify and filter valid extensions
		self.files = sorted(set(self.files))
		self.files = [
			f for f in self.files
				if str(f).lower().split(".")[-1] in EXTENSIONS_VALID
		]

		if not self.files:
			raise ValueError("No valid audio files found.")

	def _process_files(self,
		start: str = None,
		duration: str = None,
		skip_each: int = 1,

		checkpoint_path: str = None,

		n_fft: int = None,
		freq_step: int = None,
		show_graph: bool = False,

		include_samplerate: bool = True,
		include_channels: bool = False,
		include_bitrate: bool = False,
		include_peak: bool = False
	):
		outs = {}

		for file in self.files:
			with sf.SoundFile(file) as f:
				y = f.read(dtype = "float32").T
				sr = f.samplerate

				n_frames = f.frames

				subtype_string = f.subtype

				if "_" in subtype_string:
					subtype_splitted = subtype_string.split("_")

					if len(subtype_splitted):
						if len(subtype_splitted) > 1:
							bit_depth = 16 # possibly for MP3
						else:
							bit_depth = subtype_splitted[-1] # invalid on FLAC, as 24-bit do exist
					else:
						bit_depth = 32 # "FLOAT"?

				bit_depth = int(bit_depth)

			if y.size == 0:
				raise ValueError(f"{f} is an empty file")

			if not start:
				start = 0

			start = to_samples(start, sr)

			if not duration:
				duration = n_frames

			duration = to_samples(duration, sr)
			duration = max(duration, n_frames)

			if start == duration:
				raise ValueError(f"Start and duration are the same for {file}.")

			y_trimmed = y[..., start:duration:skip_each]
			y_trimmed_max = y_trimmed / (np.max(np.abs(y_trimmed)) + 1e-12)

			# for validation
			s_extended = int(float(min(SAMPLE_MIN_EXTEND, SAMPLE_LARGE)))

			if end < s_extended:
				y_trimmed_max = extend_signal(y_trimmed_max, s_extended)

			result = {}

			if include_samplerate:
				estimated_sr = Estimators.sample_rate(
					y_trimmed_max,
					sr,

					checkpoint_path = checkpoint_path,

					n_fft = n_fft,
					freq_step = freq_step,
					show_graph = show_graph,
					rounded = False
				)

				result.update({"samplerate": int(estimated_sr)})
				result.update({"cutoff": estimated_sr / 2})

			estimated_channels = Estimators.channels(y_trimmed_max)

			if include_channels:
				result.update({"channels": estimated_channels})

			if include_bitrate:
				if not include_samplerate:
					warnings.warn("Cannot include estimated bitrate without estimating sample rate")

				if not include_channels:
					warnings.warn("Cannot include estimated bitrate without estimating channels")

				if include_samplerate and include_channels:
					estimated_bitrate = Estimators.bitrate(estimated_sr, estimated_channels, bit_depth)
					result.update({"bitrate": estimated_bitrate})

			if include_peak:
				estimated_peak = Estimators.peak(y_trimmed_max)
				result.update({"peak": estimated_peak})

			outs[file] = result

		return outs

	def channels(
		self,
		start: int = None,
		duration: str = None,
		skip_each: int = 1
	) -> Dict[str, int]:
		"""
		Analyze and return the number of channels for each audio file.

		Args:
			start (str): Start time in seconds or timestamp string.
			duration (str): Duration in seconds or timestamp string.
			skip_each (int): Amount of processed samples to skip.

		Returns:
			dict[str, int]: Mapping of filename to channel count.
		"""
		return self._process_files(
			start,
			duration,
			skip_each = skip_each,

			checkpoint_path = None,

			n_fft = None,
			freq_step = None,
			show_graph = False,

			include_samplerate = False,
			include_channels = True,
			include_bitrate = False,
			include_peak = False
		)

	def sample_rate(
		self,
		start: str = None,
		duration: str = None,
		skip_each: int = 1,

		checkpoint_path: str = None,

		freq_step: int = None,
		show_graph: bool = False
	) -> Dict[str, List[int]]:
		"""
		Estimate sample rate for files with optional checkpoint path and frequency step.

		Args:
			start (str): Start time in seconds or timestamp string.
			duration (str): Duration in seconds or timestamp string.
			skip_each (int): Amount of processed samples to skip.
			checkpoint_path (str, optional): Checkpoint path.
			freq_step (int, optional): Frequency step for estimation.
			show_graph: (bool, optional): Show graph of sample rate estimation.

		Returns:
			Dict[str, List[int]]: Mapping of filename to estimated sample rate.
		"""
		return self._process_files(
			start,
			duration,
			skip_each = skip_each,

			checkpoint_path = checkpoint_path,

			n_fft = None,
			freq_step = freq_step,
			show_graph = show_graph,

			include_samplerate = True,
			include_channels = False,
			include_bitrate = False,
			include_peak = False
		)

	def estimate(
		self,
		start: str = None,
		duration: str = None,
		skip_each: int = 1,

		checkpoint_path: str = None,

		n_fft: int = None,
		freq_step: int = None,
		show_graph: bool = False,
	) -> Dict[str, Dict[str, Any]]:
		"""
		Analyze audio info: sample rate, channels, bitrate, and peak level.

		Args:
			start (str): Start time in seconds or timestamp string.
			duration (str): Duration in seconds or timestamp string.
			skip_each (int): Amount of processed samples to skip.
			checkpoint_path (str, optional): Model path or identifier.
			n_fft (int, optional): FFT window size used in heuristic sample rate estimation.
			freq_step (int, optional): Frequency step for estimation.
			show_graph: (bool, optional): Show graph of sample rate estimation.

		Returns:
			Dict[str, Dict[str, Any]]: Mapping of filename to a dict with keys:
				"samplerate", "cutoff", "channels", "bitrate", "peak".
		"""
		return self._process_files(
			start,
			duration,

			checkpoint_path = checkpoint_path,

			n_fft = None,
			freq_step = freq_step,
			show_graph = show_graph,

			include_samplerate = True,
			include_channels = True,
			include_bitrate = True,
			include_peak = True
		)

	aio = estim = estimate
	ndim = chan = channels
	sr = samplerate = sample_rate