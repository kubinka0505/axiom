from ._core.setup import (
	EXTENSIONS_VALID,
	SAMPLE_LARGE,
	SAMPLE_MIN_EXTEND,
	DEPTHS_BIT
)

from ._core.helpers.iterables import snap
from ._core.helpers.audio import (
	to_samples,
	extend_signal,
	trim_signal,
	spectral_gate
)

from ._core.helpers.paths import normalize_path
from ._core.algorithms import Estimators as Estimators

import warnings
import numpy as np
import soundfile as sf
from pathlib import Path

from typing import Any, List, Dict

#-=-=-=-#

class Axiom:
	def __init__(self, filepaths: List[str | Path], recursive: bool = False) -> None:
		"""
		Initialize with a list of file paths or directories, optionally recursively.

		Parameters
		----------
			filepaths (list[str | Path]):
				List of file or directory paths.

			recursive (bool):
				Whether to recursively search directories.

		Raises
		------
			ValueError:
				If no valid audio files are found or paths are invalid.
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

		spectral_gate_cutoff_db: float = None,
		spectral_gate_bands: list[list[float]] | None = None,
		checkpoint_path: str = None,

		n_fft: int = None,
		freq_step: int = None,
		show_graph: bool = False,

		include_samplerate: bool = True,
		include_bit_depth: bool = False,
		include_channels: bool = False,
		include_bitrate: bool = False,
		include_peak: bool = False
	):
		outs = {}

		for file in self.files:
			with sf.SoundFile(file) as f:
				sr = f.samplerate
				signal = f.read(dtype = "float32")

				if not signal.size:
					raise ValueError(f"{file} is an empty file")

				# enforce canonical shape (frames, channels)
				if signal.ndim == 1:
					signal = signal[:, None]

				n_frames = signal.shape[0]

				subtype_string = f.subtype

				bit_depth = 32

				if "_" in subtype_string:
					subtype_splitted = subtype_string.split("_")

					if len(subtype_splitted) > 1 and subtype_splitted[-1].isdigit():
						bit_depth = int(subtype_splitted[-1])

			# --- audio.py's trim/extend/spectral_gate all expect channel-first
			# --- (channels, frames), but signal here is (frames, channels).
			# --- Transpose there and back around that trio of calls.
			signal_cf = signal.T  # (n_channels, n_frames)

			# --- TRIM FIRST (correct axis: frames, now the last axis) ---
			signal_cf, _, _ = trim_signal(
				signal_cf,
				start = to_samples(start or 0, sr),
				duration = to_samples(duration, sr) if duration is not None else -1,
				skip_each = skip_each,
				max_duration = n_frames,
			)

			# --- NORMALIZE EARLY (stable estimators + gating) ---
			signal_cf = signal_cf / (np.max(np.abs(signal_cf)) + 1e-12)

			# --- EXTEND AFTER NORMALIZATION ---
			target_len = int(min(SAMPLE_MIN_EXTEND, SAMPLE_LARGE))

			if signal_cf.shape[-1] < target_len:
				signal_cf = extend_signal(signal_cf, target_len)

			# --- SPECTRAL GATE (AFTER FIXING SHAPE) ---
			if spectral_gate_cutoff_db is not None:
				signal_cf = spectral_gate(
					signal_cf,
					sr,
					cutoff = spectral_gate_cutoff_db,
					bands = spectral_gate_bands
				)

			# --- back to (frames, channels) for Estimators ---
			signal_trimmed = signal_cf.T

			result = {}

			# ---------------- SAMPLERATE ----------------
			if include_samplerate:
				estimated_sr = Estimators.sample_rate(
					signal_trimmed,
					sr,
					checkpoint_path = checkpoint_path,
					n_fft = n_fft,
					freq_step = freq_step,
					show_graph = show_graph,
					rounded = False
				)

				result.update({
					"samplerate": int(estimated_sr),
					"cutoff": estimated_sr / 2
				})

			# ---------------- BIT DEPTH ----------------
			if include_bit_depth:
				estimated_bit_depth = Estimators.bit_depth(signal_trimmed)
				estimated_bit_depth = estimated_bit_depth if estimated_bit_depth else bit_depth

				result.update({
					"bit depth": estimated_bit_depth,
					"bit depth snapped": snap(estimated_bit_depth, DEPTHS_BIT)
				})

			# ---------------- CHANNELS ----------------
			estimated_channels = Estimators.channels(signal_trimmed)

			if include_channels:
				result.update({"channels": estimated_channels})

			# ---------------- BITRATE ----------------
			if include_bitrate:
				if include_samplerate and include_bit_depth and include_channels:
					estimated_bitrate = Estimators.bit_rate(
						file,
						estimated_sr,
						estimated_channels,
						estimated_bit_depth
					)
					result.update({"bitrate": estimated_bitrate})
				else:
					warnings.warn("Bitrate estimation requires samplerate, bit depth, and channels")

			# ---------------- PEAK ----------------
			if include_peak:
				estimated_peak = Estimators.peak(signal_trimmed)
				result.update({"peak": estimated_peak})

			outs[file] = result

		return outs

	def sample_rate(
		self,
		start: str = None,
		duration: str = None,
		skip_each: int = 1,

		spectral_gate_cutoff_db: float = None,
		spectral_gate_bands: list[list[float]] | None = None,
		checkpoint_path: str = None,

		freq_step: int = None,
		show_graph: bool = False
	) -> Dict[str, List[int]]:
		"""
		Estimate sample rate for files with optional checkpoint path and frequency step.

		Parameters
		----------
			start (str):
				Start time in seconds or timestamp string.

			duration (str):
				Duration in seconds or timestamp string.

			skip_each (int):
				Amount of processed samples to skip.

			---

			checkpoint_path (str, optional):
				Checkpoint path.

			---

			freq_step (int, optional):
				Frequency step for estimation.

			show_graph: (bool, optional):
				Show graph of sample rate estimation.

		Returns
		-------
			Dict[str, List[int]]:
				Mapping of filename to estimated sample rate.
		"""
		return self._process_files(
			start,
			duration,
			skip_each = skip_each,

			spectral_gate_cutoff_db = spectral_gate_cutoff_db,
			spectral_gate_bands = spectral_gate_bands,
			checkpoint_path = checkpoint_path,

			n_fft = None,
			freq_step = freq_step,
			show_graph = show_graph,

			include_samplerate = True,
			include_bit_depth = False,
			include_channels = False,
			include_bitrate = False,
			include_peak = False
		)

	def bit_depth(
		self,
		start: int = None,
		duration: str = None,
		skip_each: int = 1
	) -> Dict[str, int]:
		"""
		Analyze and return the bit depth for each audio file.

		Parameters
		----------
			start (str):
				Start time in seconds or timestamp string.

			duration (str):
				Duration in seconds or timestamp string.

			skip_each (int):
				Amount of processed samples to skip.

		Returns
		-------
			Dict[str, int]:
				Mapping of filename to bit depth.
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
			include_bit_depth = True,
			include_channels = False,
			include_bitrate = False,
			include_peak = False
		)

	def channels(
		self,
		start: int = None,
		duration: str = None,
		skip_each: int = 1
	) -> Dict[str, int]:
		"""
		Analyze and return the number of channels for each audio file.

		Parameters
		----------
			start (str):
				Start time in seconds or timestamp string.

			duration (str):
				Duration in seconds or timestamp string.

			skip_each (int):
				Amount of processed samples to skip.

		Returns
		-------
			dict[str, int]:
				Mapping of filename to channel count.
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
			include_bit_depth = False,
			include_channels = True,
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

		Parameters
		----------
			start (str):
				Start time in seconds or timestamp string.

			duration (str):
				Duration in seconds or timestamp string.

			skip_each (int):
				Amount of processed samples to skip.

			---

			checkpoint_path (str, optional): Model path or identifier.

			---

			n_fft (int, optional):
				FFT window size used in heuristic sample rate estimation.

			freq_step (int, optional):
				Frequency step for estimation.

			show_graph: (bool, optional):
				Show graph of sample rate estimation.

		Returns
		-------
			Dict[str, Dict[str, Any]]:
				Mapping of filename to a dict with keys:
				- "samplerate",
				- "cutoff",
				- "channels",
				- "bitrate",
				- "peak".
		"""
		return self._process_files(
			start,
			duration,

			checkpoint_path = checkpoint_path,

			n_fft = n_fft,
			freq_step = freq_step,
			show_graph = show_graph,

			include_samplerate = True,
			include_bit_depth = True,
			include_channels = True,
			include_bitrate = True,
			include_peak = True
		)

	sr = samplerate = samplingrate = sampling_rate = sample_rate
	bd = bit = bits = bit_depth
	ndim = chan = chans = channels
	aio = estim = estimate