import io
import os
import librosa
import numpy as np
import soundfile as sf
from mutagen import File as mFile
from typing import Optional, Union

from .setup import (
	RATES_MP3_BIT,
	RATES_OGG_BIT,
	STEP_CLAMP_VALUE,
	DEFAULT_VALUE_FFT
)

from .helpers.numbers import clamp
from .helpers.audio import to_db

#-=-=-=-#

class Estimators:
	def sample_rate(
		signal: np.ndarray,
		sr: int,

		checkpoint_path: str,
		device: str = "cpu",

		n_fft: int = None,
		freq_step: Optional[int] = None,
		show_graph: bool = False,

		rounded: bool = True
	) -> int:
		"""
		Sample rate estimator.

		Parameters
		----------
		signal : np.ndarray
			Audio waveform, 1D or 2D (multi-channel).

		sr : int
			Original sample rate.

		checkpoint_path : str
			Model checkpoint path for prediction.

		n_fft : Optional[int]
			FFT window size used in heuristic samplerate estimation.

		device : str
			Device to run model on ("cpu" or "cuda").

		freq_step : Optional[int]
			Frequency step for heuristic cutoff heuristic.

		show_graph : bool
			Whether to show graph for heuristic method.

		rounded : bool
			Whether to return integer instead of float.

		Returns
		-------
		int
			Estimated sample rate.
		"""
		if signal.ndim == 2:
			signal = np.mean(signal, axis = 0)

		if not checkpoint_path:
			c = heuristic_cutoff(
				signal,
				sr,

				p = freq_step,
				n_fft = n_fft,

				show = show_graph
			)

			return sr if not c else 2 * (int(c) if rounded else c)

		# moved to fasten load times
		from ..ai._core.core import predict

		cutoff = predict((signal, sr), checkpoint_path, None, device)
		samplerate = int(2 * cutoff)

		if samplerate == sr:
			if "logger" in globals():
				logger.error(Fore.RED + "Model returned input sample rate, reverting to heuristic estimation." + Fore.RESET)

			c = heuristic_cutoff(
				signal,
				sr,

				p = freq_step,
				n_fft = n_fft,

				show = show_graph
			)
			retval = sr if not retval else 2 * (int(c) if rounded else c)
		else:
			retval = samplerate

		if not retval:
			retval = sr

		return retval

	def bit_depth(
		signal: np.ndarray,

		min_depth: int = 8,
		max_depth: int = 32,

		return_details: bool = False
	) -> Union[int, tuple[int | None, dict], None]:
		"""
		Estimate effective bit depth of an audio signal. (ENOB)

		This is a heuristic combining:
		- quantization error (MSE)
		- SNR-based estimation
		- LSB / histogram structure

		Returns
		-------
		Union[int, tuple[int | None, dict], None]
			Estimated bit depth with diagnostics if `return_details` is True
		"""
		# preprocess
		signal = np.asarray(signal)

		if signal.ndim > 1:
			signal = np.mean(signal, axis = 1)

		signal = signal.astype(np.float64)

		# normalize (important for comparability)
		peak = np.max(np.abs(signal)) + 1e-12
		signal = np.clip(signal / peak, -1, 1)

		signal_power = np.mean(signal ** 2) + 1e-12

		# storage
		results = []

		# scan bit depths
		for depth in range(min_depth, max_depth + 1):
			max_int = 2 ** (depth - 1) - 1

			quant = np.round(signal * max_int) / max_int

			noise = signal - quant

			noise_power = np.mean(noise ** 2) + 1e-12

			snr = 10 * np.log10(signal_power / noise_power)

			mse = noise_power

			# expected SNR ~ 6.02 * bits + 1.76 (ideal PCM)
			expected_snr = 6.02 * depth + 1.76

			snr_error = abs(snr - expected_snr)

			results.append((depth, mse, snr, snr_error))

		# choose best candidate
		# prioritize:
		# - low SNR mismatch (most important)
		# - then low noise

		best = min(results, key = lambda x: (x[3], x[1]))
		best_depth = best[0]

		# optional LSB structure check
		# detects real quantization steps (PCM-like signals)

		## Heuristic: detects whether signal has discrete quantization levels.
		# 0.0 -> no visible quantization structure (noise-like / dithered)
		# 1.0 -> strong PCM-like step structure

		# difference signal reveals quantization steps
		signal_diff = np.diff(signal)

		if len(signal_diff) < 10:
			hist_score = 0

		# histogram of step sizes
		hist, _ = np.histogram(signal_diff, bins = 100, density = True)

		# entropy: low entropy => structured quantization
		hist = hist + 1e-12
		entropy = -np.sum(hist * np.log(hist))

		# normalize entropy into 0..1 score
		hist_score = 1 - (entropy / np.log(len(hist)))
		hist_score = float(np.clip(hist_score, 0, 1))

		# adjust estimate slightly if strong quantization structure exists
		if hist_score > 0.7:
			best_depth = min(best_depth + 1, max_depth)

		# output
		if return_details:
			details = {
				"best_snr": best[2],
				"snr_error": best[3],
				"lsb_score": hist_score,
				"all_results": results,
			}

			return best_depth, details

		return best_depth

	def channels(
		signal: np.ndarray,
		chunk_size: int = 2048,
		stereo_threshold: float = 0.05
	) -> int:
		"""
		Determine number of audio channels with higher precision.
		Uses phase cancellation via pydub if available, otherwise falls back to chunk-based analysis.

		Parameters
		----------
		signal : np.ndarray
			1D or 2D array.

		chunk_size : int
			Number of samples per analysis chunk (fallback mode).

		stereo_threshold : float
			Fraction of chunks that must show stereo difference (fallback mode).

		Returns
		-------
		int
			Number of channels (1 for mono, 2 for stereo, or more).
		"""
		# mono
		if signal.ndim == 1:
			return 1

		# ensure shape (samples, channels)
		if signal.shape[0] < signal.shape[1]:
			signal = signal.T

		# return channel count directly
		if signal.shape[1] != 2:
			return signal.shape[1]

		try:
			from pydub import AudioSegment
		except ImportError:
			return Estimators._channels_chunking(signal, chunk_size, stereo_threshold)

		signal_scaled = np.int16(
			np.clip(signal, -1, 1) * 32767
		)

		seg = AudioSegment(
			signal_scaled.tobytes(),
			frame_rate = 44100,
			sample_width = 2,
			channels = 2
		)

		# Split channels
		left = seg.split_to_mono()[0]
		right = seg.split_to_mono()[1]

		left_inverted = left.invert_phase()
		diff = right.overlay(left_inverted)

		if diff.max_dBFS > -80.0:
			return 2

		return 1

	def _channels_chunking(
		signal: np.ndarray,
		chunk_size: int,
		stereo_threshold: float
	) -> int:
		"""
		Fallback method for chunk-based stereo detection.
		"""
		left, right = signal[:, 0], signal[:, 1]
		signal = np.clip(signal * 10, -1, 1)

		stereo_chunks = 0
		total_chunks = 0

		for start in range(0, len(left), chunk_size):
			end = min(start + chunk_size, len(left))
			l_chunk = left[start:end]
			r_chunk = right[start:end]

			if len(l_chunk) < 2:
				continue

			if np.std(l_chunk) == 0 or np.std(r_chunk) == 0:
				corr = 1
			else:
				corr = np.corrcoef(l_chunk, r_chunk)[0, 1]

			diff = np.mean(np.abs(l_chunk - r_chunk))

			if corr < 0.999 and diff >= 1e-6:
				stereo_chunks += 1

			total_chunks += 1

		if total_chunks > 0 and (stereo_chunks / total_chunks) >= stereo_threshold:
			return 2

		return 1

	def bit_rate(
		file: str,

		sr: Optional[int] = None,
		n_channels: Optional[int] = None,
		bit_depth: Optional[int] = None
	) -> int:
		"""
		Compute bitrate from sample rate, channels, and bit depth.

		Parameters
		----------
		file : str
			Input file path.

		sr : Optional[int]
			Sample rate in Hz.

		n_channels : Optional[int]
			Number of audio channels.

		bit_depth : Optional[int]
			Bit depth per sample. Ommited in FLAC files.

		Returns
		-------
		int
			Bitrate in bits per second.
		"""
		bitrate = sr * n_channels * bit_depth
		audio_fmt = os.path.splitext(file)[-1].strip(".").upper()

		if audio_fmt == "WAV":
			return bitrate

		try:
			from pydub import AudioSegment
			AVAILABLE_PYDUB = True
		except ImportError:
			AVAILABLE_PYDUB = False

		if AVAILABLE_PYDUB and audio_fmt == "FLAC":
			mFileObj = mFile(file)
			audio = AudioSegment.from_file(file)

			if n_channels:
				audio = audio.set_channels(n_channels)

			buf = io.BytesIO()
			export_params = [
				"-map_metadata", "-1",
				"-compression_level", "8",
				"-fflags", "+bitexact",
				"-flags:v", "+bitexact",
				"-flags:a", "+bitexact"
			]

			if sr:
				export_params.extend(["-af", f"aresample={sr}:cutoff=1"])

			audio.export(buf, format = "flac", parameters = export_params)
			buf.seek(0)

			# buf = optimize_file_audio(buf)

			with sf.SoundFile(buf) as f:
				frames = len(f)
				samplerate = f.samplerate

				duration_sec = frames / samplerate
				buf.seek(0, io.SEEK_END)
				size_bytes = buf.tell()
				bitrate = (size_bytes * 8) / duration_sec # true bitrate in bps

			# cba
			orig_rate = mFileObj.info.bitrate

			if orig_rate > bitrate and "logger" in globals():
				logger.debug(
					"Bitrate inconsistency detected: {0} > {1} (original)".format(
						round(orig_rate / 1000, 2),
						round(bitrate / 1000, 2)
					)
				)

			return min(bitrate, mFileObj.info.bitrate)

		if audio_fmt == "MP3":
			bitrate = clamp(bitrate, min(RATES_MP3_BIT), max(RATES_MP3_BIT))

		if audio_fmt == "OGG":
			bitrate = clamp(bitrate, min(RATES_OGG_BIT), max(RATES_OGG_BIT))

		return bitrate

	def peak(
		signal: np.ndarray,
		unit: str = "dB",
		rounding: int = 5
	) -> float:
		"""
		Get the peak level of a signal.

		Parameters
		----------
		signal : np.ndarray
			Audio signal. Can be mono (1D) or multichannel (ND, last axis is samples).

		unit : str
			"db" for decibels full scale (dBFS), "linear" for absolute amplitude.

		rounding : int
			Return value decimal precision.

		Returns
		-------
		float
			Peak level. If unit starts with "dB", returns peak in dBFS (<= 0.0, 0.0 is full scale).
			Otherwise if starts with "lin", returns max absolute sample value.

		Raises
		------
		ValueError
			If unit is not one of "db" or "linear".
		"""
		# collapse channels if multichannel
		peak_linear = np.max(np.abs(signal))

		lower = unit.lower()

		if lower.startswith("lin"):
			retval = float(peak_linear)
		elif lower.startswith("db"):
			if not peak_linear:
				retval = -float("inf")

			retval = to_db(peak_linear)
		else:
			raise ValueError("Unsupported unit.")

		if abs(retval) != float("inf"):
			return round(retval, rounding)

		return None

#-=-=-=-#

def transform_contrast(
	img:
	np.ndarray, value: float
) -> np.ndarray:
	"""
	Adjust contrast of an image represented as a float array in [0,1].

	Parameters
	----------
	img : np.ndarray
		Input image array with values in [0,1].
	value : float
		Contrast adjustment value in [-127, 127].

	Returns
	-------
	np.ndarray
		Contrast-adjusted image clipped to [0,1].
	"""
	c = np.clip(value, -127, 127)
	factor = (259 * (c + 255)) / (255 * (259 - c))
	img_adj = factor * (img - 0.5) + 0.5

	return np.clip(img_adj, 0, 1)

def compute_contrasted_spectrogram(
	signal: np.ndarray,
	sr: int,
	contrast: float = 127.0,
	n_fft: int = DEFAULT_VALUE_FFT,
	hop_length: Optional[int] = None,
	power: float = 2.0
) -> np.ndarray:
	"""Helper to compute normalized and contrast-adjusted magnitude spectrogram."""
	if hop_length is None or hop_length <= 0:
		hop_length = n_fft // 4

	signal = np.asarray(signal, dtype = np.float32)
	if signal.ndim == 2:
		signal = np.mean(signal, axis = 0)

	# compute magnitude STFT and convert to dB
	stft = librosa.stft(signal, n_fft = n_fft, hop_length = hop_length, window = "hann")
	s_mag = np.abs(stft) ** power
	s_db = librosa.power_to_db(s_mag, ref = np.max)

	# normalize to [0, 1] with fallback for silent audio
	db_range = s_db.max() - s_db.min()
	if db_range < 1e-6:
		s_norm = np.zeros_like(s_db)
	else:
		s_norm = (s_db - s_db.min()) / db_range

	return transform_contrast(s_norm, contrast)

def detect_cutoff_index(
	vertical_profile_flipped: np.ndarray,
	step_size: int,
	threshold: float = 1e-3
) -> Optional[int]:
	"""Fast non-visual top-down scanning algorithm."""
	height = len(vertical_profile_flipped)
	idx = 0

	while idx < height:
		if vertical_profile_flipped[idx] > threshold:
			# Binary search backtrack to find precise onset edge
			back_step = step_size / 2
			refined_idx = float(idx)

			while back_step >= 1:
				candidate = int(round(refined_idx - back_step))

				if candidate >= 0 and vertical_profile_flipped[candidate] > threshold:
					refined_idx = float(candidate)

				back_step /= 2

			return int(round(refined_idx))

		idx += step_size

	return None

def heuristic_cutoff(
	signal: np.ndarray,
	sr: int,

	contrast: int = 127,
	p: Optional[int] = None,

	n_fft: Optional[int] = None,
	hop_length: Optional[int] = None,

	power: int = 2,

	show: bool = False,
	output_dir: Optional[str] = None
) -> Union[float, None]:
	"""
	Estimate cutoff frequency by scanning the contrast-enhanced spectrogram from top.

	Parameters
	----------
	signal : np.ndarray
		Audio time series.
	sr : int
		Sampling rate in Hz.

	contrast : int
		Contrast level in range [-127, 127].
	p : Optional[int]
		Initial vertical scan step in pixels.

	n_fft : Optional[int]
		FFT size (controls vertical frequency resolution).
	hop_length : Optional[int]
		Hop length for STFT.

	power : int
		Power for magnitude spectrogram (2 for power, 1 for energy).

	show : bool
		If True, renders an animated scan visualization.
	output_dir : Optional[str]
		Directory to export frame PNGs if provided.

	Returns
	-------
	Union[float, None]
		Estimated cutoff frequency in Hz.
	"""
	n_fft = n_fft or DEFAULT_VALUE_FFT

	# resolve step size parameter p
	default_p = max(1, int(sr / STEP_CLAMP_VALUE) - 1)

	if p is None or p <= 0:
		step_size = default_p
	else:
		step_size = max(1, min(p, default_p))

	# Process spectrogram
	spectrogram_contrasted = compute_contrasted_spectrogram(
		signal, sr,
		contrast = contrast,
		n_fft = n_fft,
		hop_length = hop_length,
		power = power
	)

	# max-pool along time dimension to get frequency energy envelope
	vertical_profile = np.max(spectrogram_contrasted, axis = 1)
	vertical_profile_flipped = vertical_profile[::-1]

	height = len(vertical_profile_flipped)
	nyq = sr / 2
	freq_bin_width = nyq / max(1, height - 1)

	# Fast path: compute result directly without GUI loop if show is False
	if not show:
		found_idx = detect_cutoff_index(vertical_profile_flipped, step_size)
		return (nyq - found_idx * freq_bin_width) if found_idx is not None else None

	import matplotlib as mpl
	import matplotlib.pyplot as plt

	# Visualization path
	font = "Arial"
	mpl.rcParams["savefig.dpi"] = 200
	mpl.rcParams["figure.dpi"] = 100
	mpl.rcParams["font.family"] = "monospace"
	mpl.rcParams["font.size"] = 11

	plt.ion()
	fig, ax = plt.subplots(figsize = (8, 6))

	ax.imshow(
		spectrogram_contrasted[::-1, :],
		aspect = "auto",
		origin = "upper",
		interpolation = "nearest",
		cmap = "magma"
	)

	ax.set_title(f"{nyq:.2f} Hz", fontsize = 13, fontname = font)
	yticks_idx = np.linspace(0, height - 1, min(5, height), dtype = int)
	yticks_freq = nyq - yticks_idx * freq_bin_width

	ax.set_yticks(yticks_idx)
	ax.set_yticklabels([f"{f:.0f}" for f in yticks_freq])
	ax.set_xticks([0, spectrogram_contrasted.shape[1] - 1])
	ax.set_xticklabels(["0", str(len(signal))])
	ax.set_ylabel("Frequency (Hz)", fontname = font)
	ax.set_xlabel("Sample index", fontname = font)

	scan_line = ax.axhline(0, color = "cyan", linewidth = 1, label = "scan")

	if output_dir:
		os.makedirs(output_dir, exist_ok = True)

	frame_count = 1
	idx = 0
	found_idx = None
	eps = 1e-3

	while idx < height:
		cutoff_hz = nyq - idx * freq_bin_width
		ax.set_title(f"{cutoff_hz:.2f} Hz")
		scan_line.set_ydata([idx, idx])

		fig.canvas.draw_idle()
		plt.pause(0.005)

		if output_dir:
			fig.savefig(os.path.join(output_dir, f"frame_{frame_count:04d}.png"), facecolor = "white")
			frame_count += 1

		if vertical_profile_flipped[idx] > eps:
			back_step = step_size / 2
			refined_idx = float(idx)

			while back_step >= 1:
				candidate = int(round(refined_idx - back_step))

				if candidate >= 0 and vertical_profile_flipped[candidate] > eps:
					refined_idx = float(candidate)

				back_step /= 2

			found_idx = int(round(refined_idx))
			scan_line.set_ydata([found_idx, found_idx])
			scan_line.set_color("lime")
			scan_line.set_linewidth(2)

			cutoff_hz = nyq - found_idx * freq_bin_width
			ax.set_title(f"{cutoff_hz:.2f} Hz")
			fig.canvas.draw_idle()
			plt.pause(0.01)

			if output_dir:
				fig.savefig(os.path.join(output_dir, f"frame_{frame_count:04d}.png"), facecolor = "white")

			break

		idx += step_size

	plt.ioff()
	plt.pause(1)
	plt.close(fig)

	return (nyq - found_idx * freq_bin_width) if found_idx is not None else None