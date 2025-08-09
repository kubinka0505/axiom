import math
import warnings

import librosa
import numpy as np

from typing import Optional

from ._setup import *

warnings.filterwarnings("error")

#-=-=-=-#

class Estimators:
	def sample_rate(
		y: np.ndarray,
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

		Args:
			y (np.ndarray): Audio waveform, 1D or 2D (multi-channel).
			sr (int): Original sample rate.
			checkpoint_path (str): Model checkpoint path for prediction.
			n_fft (int): FFT window size used in heuristic samplerate estimation.
			device (str): Device to run model on ("cpu" or "cuda").
			freq_step (Optional[int]): Frequency step for adaptive cutoff heuristic.
			show_graph (bool): Whether to show graph for heuristic method.
			rounded (bool): Whether to return integer instead of float.

		Returns:
			int: Estimated sample rate.
		"""
		if y.ndim == 2:
			y = np.mean(y, axis = 0)

		if not checkpoint_path:
			c = adaptive_cutoff(
				y,
				sr,
				p = freq_step,
				n_fft = n_fft,
				show = show_graph
			)
			return sr if not c else 2 * (int(c) if rounded else c)

		cutoff = predict_cutoff((y, sr), checkpoint_path, device)
		samplerate = int(2 * cutoff)

		if samplerate == sr:
			if "logger" in globals():
				logger.error(Fore.RED + "Model returned input sample rate, reverting to heuristic estimation." + Fore.RESET)

			c = adaptive_cutoff(
				y,
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

	def bitrate(sr: int, n_channels: int, bit_depth: int) -> int:
		"""
		Compute bitrate from sample rate, channels, and bit depth.

		Args:
			sr (int): Sample rate in Hz.
			n_channels (int): Number of audio channels.
			bit_depth (int): Bit depth per sample.

		Returns:
			int: Bitrate in bits per second.
		"""
		return sr * bit_depth * n_channels

	def channels(y: np.ndarray, chunk_size: int = 2048, stereo_threshold: float = 0.05) -> int:
		"""
		Determine number of audio channels in the signal with higher precision.
		Works even if stereo is present only in small sections.

		Args:
			y (np.ndarray): Audio waveform, 1D or 2D array.
			chunk_size (int): Number of samples per analysis chunk.
			stereo_threshold (float): Fraction of chunks that must show stereo difference to count as stereo.

		Returns:
			int: Number of channels (1 for mono, 2 for stereo, or more).
		"""
		# mono signal
		if y.ndim == 1:
			return 1

		# ensure shape (samples, channels)
		if y.shape[0] < y.shape[1]:
			y = y.T

		# if not mono return directly
		if y.shape[1] != 2:
			return y.shape[1]

		left, right = y[:, 0], y[:, 1]

		# clip to [-1, 1] range after boosting to avoid low-amplitude rounding issues
		y = np.clip(y * 10, -1.0, 1.0)

		stereo_chunks = 0
		total_chunks = 0

		# process in chunks
		for start in range(0, len(left), chunk_size):
			end = min(start + chunk_size, len(left))

			l_chunk = left[start:end]
			r_chunk = right[start:end]

			if len(l_chunk) < 2:
				continue # skip too short

			# correlation & mean difference for this chunk
			corr = np.corrcoef(l_chunk, r_chunk)[0, 1]
			diff = np.mean(np.abs(l_chunk - r_chunk))

			# consider this chunk stereo if difference is meaningful
			if corr < 0.999 and diff >= 1e-5:
				stereo_chunks += 1

			total_chunks += 1

		# if enough chunks are stereo, call the whole signal stereo
		if total_chunks > 0 and (stereo_chunks / total_chunks) >= stereo_threshold:
			return 2
		else:
			return 1

	def peak(y: np.ndarray, unit: str = "dB", rounding: int = 5) -> float:
		"""
		Get the peak level of a signal.

		Args:
			y (np.ndarray): Audio signal. Can be mono (1D) or multichannel (ND, last axis is samples).
			unit (str): "db" for decibels full scale (dBFS), "linear" for absolute amplitude.
			rounding (int): Return value decimal precision.

		Returns:
			float: Peak level. If unit starts with "dB", returns peak in dBFS (<= 0.0, 0.0 is full scale).
				Otherwise if starts with "lin", returns max absolute sample value.

		Raises:
			ValueError: If unit is not one of "db" or "linear".
		"""
		# collapse channels if multichannel
		# (consider the maximum across all samples/channels)
		peak_linear = np.max(np.abs(y))

		lower = unit.lower()

		if lower.startswith("lin"):
			retval = float(peak_linear)
		elif lower.startswith("db"):
			if not peak_linear:
				retval = -float("inf")

			retval = 20.0 * np.log10(peak_linear)
		else:
			raise ValueError("Unsupported unit.")

		if abs(retval) != float("inf"):
			return round(retval, rounding)

		return None

#-=-=-=-#

def transform_contrast(img: np.ndarray, value: float) -> np.ndarray:
	"""
	Adjust contrast of an image represented as a float array in [0,1].

	Args:
		img (np.ndarray): Input image array with values in [0,1].
		value (float): Contrast adjustment value in [-127, 127].

	Returns:
		np.ndarray: Contrast-adjusted image clipped to [0,1].
	"""
	C = np.clip(value, -127, 127)
	factor = (259 * (C + 255)) / (255 * (259 - C))
	img_adj = factor * (img - 0.5) + 0.5

	return np.clip(img_adj, 0.0, 1.0)

def adaptive_cutoff(
	y: np.ndarray,
	sr: int,
	contrast: int = 127,
	p: int = None,
	n_fft: int = None,
	hop_length: int = None,
	power: int = 2,
	show: bool = False
):
	"""
	Estimate cutoff frequency by scanning the contrast-enhanced spectrogram from top.
	
	Args:
		y: audio time series
		sr: sampling rate
		contrast: integer [-127, 127].
		p: initial vertical step in pixels. Truncated to `(sr / 800) - 1`
		n_fft: FFT size (controls vertical resolution)
		hop_length: hop length for STFT; defaults to n_fft // 4 if None
		power: power for magnitude spectrogram (2 for power)

	Returns:
		cutoff_hz: approximate cutoff frequency in Hz
	"""
	import matplotlib as mpl
	import matplotlib.pyplot as plt

	time_wait = 1.25

	if not p or p < 0:
		p = int(sr / 800) - 1

	p = min(int(sr / 800 - 1), sr)

	if not hop_length or hop_length < 0:
		hop_length = n_fft // 4

	y = np.asarray(y)

	if y.ndim == 2:
		y = np.mean(y, axis = 0)

	# compute magnitude spectrogram (power)
	S = librosa.stft(y, n_fft = n_fft, hop_length = hop_length, window = "hann")

	# power spectrogram
	S_mag = np.abs(S) ** power

	# convert to dB to compress dynamic range - more visually meaningful
	S_db = librosa.power_to_db(S_mag, ref = np.max)

	# normalize to [0, 1]
	S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-10)

	# apply contrast
	spectrogram_contrasted = transform_contrast(S_norm, contrast)

	# treat each column equally
	# scan per vertical axis only
	# collapse time by max to emphasize cutoff edge
	# using maximum over time so persistent high-energy frequency shows
	vertical_profile = np.max(spectrogram_contrasted, axis = 1)

	# librosa’s stft has frequency bin 0 = DC (lowest), last = Nyquist
	# now index 0 is Nyquist
	vertical_profile_flipped = vertical_profile[::-1]

	# number of frequency bins; top is highest frequency (librosa uses 0 at bottom)
	height = vertical_profile_flipped.shape[0]

	# convert to a "pixel image" where black is near zero. defined threshold for "non-black".
	eps = 1e-3

	# Visualization setup
	if show:
		font = "Arial"

		mpl.rcParams["savefig.dpi"] = 200
		mpl.rcParams["figure.dpi"] = int(mpl.rcParams["savefig.dpi"] / 2)

		mpl.rcParams["font.family"] = "monospace"
		mpl.rcParams["font.size"] = 11

		mpl.rcParams["figure.facecolor"] = "none"
		mpl.rcParams["axes.facecolor"] = "none"

		plt.ion()
		fig, ax = plt.subplots(figsize = (8, 6))
		fig.canvas.manager.set_window_title("Adaptive cutoff scan")

		ax.imshow(
			spectrogram_contrasted[::-1, :],
			aspect = "auto",
			origin = "upper",
			interpolation = "nearest",
			cmap = "magma"
		)

		scan_line = ax.axhline(0, color = "cyan", linewidth = 1, label = "scan")
		edge_line = ax.axhline(0, color = "lime", linewidth = 2, label = "detected edge")

		ax.set_title("Contrast spectrogram (top = Nyquist)", fontsize = 13, fontname = font)

		# Set vertical axis ticks (frequency)
		nyq = sr / 2.0

		height = spectrogram_contrasted.shape[0]
		freq_bin_width = nyq / (height - 1)

		num_vticks = min(5, height)
		yticks_idx = np.linspace(0, height - 1, num_vticks, dtype = int)
		yticks_freq = nyq - yticks_idx * freq_bin_width

		ax.set_yticks(yticks_idx)
		ax.set_yticklabels([f"{f:.0f}" for f in yticks_freq])

		# Set horizontal axis ticks (samples)
		ax.set_xticks([0, spectrogram_contrasted.shape[1] - 1])
		ax.set_xticklabels(["0", str(y.shape[0])])

		ax.set_ylabel("Frequency (Hz)", fontname = font)
		ax.set_xlabel("Sample index", fontname = font)

		fig.canvas.draw()
		fig.canvas.flush_events()

	# heuristic scanning from top every p pixels
	idx = 0
	found_idx = None

	while idx < height:
		val = vertical_profile_flipped[idx]

		if show:
			xd = scan_line.get_xdata()
			scan_line.set_ydata([idx] * len(xd))
			fig.canvas.draw_idle()
			plt.pause(0.005)

		if val > eps:
			# backtrack with halved step until black pixel is met, then take previous non-black
			back_step = p / 2.0

			# work in float so it can halve; index needs int
			refined_idx = idx

			while back_step >= 1:
				candidate = int(round(refined_idx - back_step))

				if candidate < 0:
					break
				if vertical_profile_flipped[candidate] > eps:
					refined_idx = candidate
				back_step /= 2.0
	
			found_idx = int(round(refined_idx))

			if show:
				xd = edge_line.get_xdata()
				edge_line.set_ydata([found_idx] * len(xd))

				# compute cutoff frequency for title
				nyq = sr / 2.0
				freq_bin_width = nyq / (height - 1)
				cutoff_hz = nyq - found_idx * freq_bin_width
				ax.set_title(f"{cutoff_hz} Hz") #" | plot will close in {time_wait}s")

				fig.canvas.draw_idle()
				plt.pause(0.01)
			break

		idx += int(round(p))

	if show:
		plt.ioff()
		plt.show(block = False)
		plt.pause(time_wait)
		plt.close(fig)

	# no edge detected; return None or zer
	if found_idx is None:
		return None

	# in hz
	nyq = sr / 2.0

	# since flipped, index 0 -> nyq, index height-1 -> ~0
	freq_bin_width = nyq / (height - 1)

	# because of flipping
	cutoff_hz = nyq - found_idx * freq_bin_width

	return cutoff_hz

def resample_audio(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
	"""
	Resample audio data to a target sample rate using polyphase filtering.

	Args:
		y (np.ndarray): Input audio array. Can be mono or multichannel.
		orig_sr (int): Original sample rate.
		target_sr (int): Target sample rate.

	Returns:
		np.ndarray: Resampled audio array.

	Raises:
		ValueError: If `orig_sr` or `target_sr` are invalid.
	"""
	# determine up/down sampling ratio
	from scipy.signal import resample_poly

	factor = math.gcd(orig_sr, target_sr)
	up = target_sr // factor
	down = orig_sr // factor

	# resample each channel independently
	if y.ndim == 2:
		resampled = np.vstack([
			resample_poly(y[ch], up, down)
			for ch in range(y.shape[0])
		])
	else:
		resampled = resample_poly(y, up, down)

	return resampled