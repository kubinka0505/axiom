import os
import math
import warnings
import soundfile as sf
from mutagen import File as mFile

import librosa
import numpy as np

from typing import Optional

from ._setup import *
from ._helper import clamp

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
			freq_step (Optional[int]): Frequency step for heuristic cutoff heuristic.
			show_graph (bool): Whether to show graph for heuristic method.
			rounded (bool): Whether to return integer instead of float.

		Returns:
			int: Estimated sample rate.
		"""
		if y.ndim == 2:
			y = np.mean(y, axis = 0)

		if not checkpoint_path:
			c = heuristic_cutoff(
				y,
				sr,

				p = freq_step,
				n_fft = n_fft,

				show = show_graph
			)
			return sr if not c else 2 * (int(c) if rounded else c)

		cutoff = predict_cutoff((y, sr), checkpoint_path, None, device)
		samplerate = int(2 * cutoff)

		if samplerate == sr:
			if "logger" in globals():
				logger.error(Fore.RED + "Model returned input sample rate, reverting to heuristic estimation." + Fore.RESET)

			c = heuristic_cutoff(
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

	def bit_depth(y: np.ndarray, max_depth: int = 24, threshold: float = 1E-6) -> int | None:
		"""
		Roughly estimate bit depth.

		Args:
			y (np.ndarray): mono or stereo float signal (-1..1).
			max_depth (int): maximum bit depth to test.
			threshold (float): acceptable reconstruction error.

		Returns:
			Estimated bit depth (int), or None if inconclusive.
		"""
		if y.ndim > 1:
			y = np.mean(y, axis = 0)

		y = np.clip(y, -1, 1)

		for depth in range(8, max_depth + 1):
			scale = 2 ** (depth - 1) - 1
			quantized = np.round(y * scale) / scale

			err = np.mean((y - quantized) ** 2)

			if err < threshold:
				return depth

		return None

	def channels(y: np.ndarray, chunk_size: int = 2048, stereo_threshold: float = 0.05) -> int:
		"""
		Determine number of audio channels with higher precision.
		Uses phase cancellation via pydub if available, otherwise falls back to chunk-based analysis.

		Args:
			y (np.ndarray): 1D or 2D array.
			chunk_size (int): Number of samples per analysis chunk (fallback mode).
			stereo_threshold (float): Fraction of chunks that must show stereo difference (fallback mode).

		Returns:
			int: Number of channels (1 for mono, 2 for stereo, or more).
		"""
		# mono
		if y.ndim == 1:
			return 1

		# ensure shape (samples, channels)
		if y.shape[0] < y.shape[1]:
			y = y.T

		# return channel count directly
		if y.shape[1] != 2:
			return y.shape[1]

		try:
			from pydub import AudioSegment
		except ImportError:
			return _channels_chunking(y, chunk_size, stereo_threshold)

		y_int16 = np.int16(np.clip(y, -1.0, 1.0) * 32767)
		seg = AudioSegment(
			y_int16.tobytes(),
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
		else:
			return 1

	def _channels_chunking(y: np.ndarray, chunk_size: int, stereo_threshold: float) -> int:
		"""
		Fallback method: chunk-based stereo detection.
		"""
		left, right = y[:, 0], y[:, 1]
		y = np.clip(y * 10, -1.0, 1.0)

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

			if corr < 0.999 and diff >= 1E-6:
				stereo_chunks += 1

			total_chunks += 1

		if total_chunks > 0 and (stereo_chunks / total_chunks) >= stereo_threshold:
			return 2
		else:
			return 1

	def bit_rate(
		file: str,

		sr: Optional[int] = None,
		n_channels: Optional[int] = None,
		bit_depth: Optional[int] = None
	) -> int:
		"""
		Compute bitrate from sample rate, channels, and bit depth.

		Args:
			file (str): Input file.
			sr (Optional[int]): Sample rate in Hz.
			n_channels (Optional[int]): Number of audio channels.
			bit_depth (Optional[int]): Bit depth per sample. Ommited in FLAC files.

		Returns:
			int: Bitrate in bits per second.
		"""
		bitrate = sr * n_channels * bit_depth
		audio_fmt = os.path.splitext(file)[-1].strip(".").upper()

		if audio_fmt == "WAV":
			return bitrate

		try:
			import io
			from pydub import AudioSegment

			AVAILABLE_PYDUB = True
		except ImportError:
			AVAILABLE_PYDUB = False

		if AVAILABLE_PYDUB:
			if audio_fmt == "FLAC":
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

				with sf.SoundFile(buf) as f:
					frames = len(f)
					samplerate = f.samplerate

					duration_sec = frames / samplerate
					buf.seek(0, io.SEEK_END)
					size_bytes = buf.tell()
					bitrate = (size_bytes * 8) / duration_sec # true bitrate in bps

				# cba
				orig_rate = mFileObj.info.bitrate

				if orig_rate > bitrate:
					if "logger" in globals():
						logger.debug("FLAC bitrate inconsistency detected: {0} > {1} (original)".format(
							round(orig_rate / 1000, 2),
							round(bitrate / 1000, 2)
						))

				return min(bitrate, mFileObj.info.bitrate)

		if audio_fmt == "MP3":
			bitrate = clamp(bitrate, min(RATES_MP3_BIT), max(RATES_MP3_BIT))

		if audio_fmt == "OGG":
			bitrate = clamp(bitrate, min(RATES_OGG_BIT), max(RATES_OGG_BIT))

		return bitrate

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

def heuristic_cutoff(
	y: np.ndarray,
	sr: int,

	contrast: int = 127,
	p: int = None,

	n_fft: int = None,
	hop_length: int = None,

	power: int = 2,
	show: bool = False
):
	f"""
	Estimate cutoff frequency by scanning the contrast-enhanced spectrogram from top.

	Args:
		y: audio time series
		sr: sampling rate
		contrast: integer [-127, 127].
		p: initial vertical step in pixels. Truncated to `(sr / {STEP_CLAMP_VALUE}) - 1`
		n_fft: FFT size (controls vertical resolution)
		hop_length: hop length for STFT; defaults to `n_fft // 4` if None
		power: power for magnitude spectrogram (2 for power)

	Returns:
		cutoff_hz: approximate cutoff frequency in Hz
	"""
	import matplotlib as mpl
	from random import randint
	import matplotlib.pyplot as plt

	if not n_fft:
		n_fft = DEFAULT_VALUE_FFT

	time_wait = 1.25

	if not p or p < 0:
		p = int(sr / STEP_CLAMP_VALUE) - 1

	p = min(int(sr / STEP_CLAMP_VALUE - 1), sr)

	if not hop_length or hop_length < 0:
		hop_length = n_fft // 4

	y = np.asarray(y)

	if y.ndim == 2:
		y = np.mean(y, axis = 0)

	# compute magnitude spectrogram (power)
	S = librosa.stft(y, n_fft = n_fft, hop_length = hop_length, window = "hann")

	# power spectrogram
	S_mag = np.abs(S) ** power

	# convert to dB
	S_db = librosa.power_to_db(S_mag, ref = np.max)

	# normalize to [0, 1]
	S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1E-10)

	# apply contrast
	spectrogram_contrasted = transform_contrast(S_norm, contrast)

	# collapse time by max to emphasize cutoff edge
	vertical_profile = np.max(spectrogram_contrasted, axis = 1)

	# flip so index 0 is Nyquist
	vertical_profile_flipped = vertical_profile[::-1]

	height = vertical_profile_flipped.shape[0]
	eps = 1E-3
	nyq = sr / 2
	freq_bin_width = nyq / (height - 1)

	# visualization setup
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
		fig.canvas.manager.set_window_title("Heuristic Cutoff Scan " + str(randint(10000, 99999)))

		ax.imshow(
			spectrogram_contrasted[::-1, :],
			aspect = "auto",
			origin = "upper",
			interpolation = "nearest",
			cmap = "magma"
		)

		ax.set_title(f"{nyq:.2f} Hz", fontsize = 13, fontname = font)

		num_vticks = min(5, height)
		yticks_idx = np.linspace(0, height - 1, num_vticks, dtype = int)
		yticks_freq = nyq - yticks_idx * freq_bin_width

		ax.set_yticks(yticks_idx)
		ax.set_yticklabels([f"{f:.0f}" for f in yticks_freq])
		ax.set_xticks([0, spectrogram_contrasted.shape[1] - 1])
		ax.set_xticklabels(["0", str(y.shape[0])])
		ax.set_ylabel("Frequency (Hz)", fontname = font)
		ax.set_xlabel("Sample index", fontname = font)

		fig.canvas.draw()
		fig.canvas.flush_events()

	# frame save setup
	dst = "_AXIOM_FRAMES"
	prefix = "frame"
	ext = "png"

	if show and os.path.exists(dst):
		for file in os.listdir(dst):
			if file.lower().startswith(prefix.lower()) and file.lower().endswith(ext.lower()):
				try:
					os.remove(os.path.join(dst, file))
				except OSError:
					pass

	def _save_fig(plot, dst: str, frame_count: int):
		if show and os.path.exists(dst):
			plot.savefig(
				os.path.join(dst, f"{prefix}_{frame_count:04d}.{ext}"),
				facecolor = "white",
				transparent = False
			)

	idx = 0
	found_idx = None
	frame_count = 1

	if show:
		scan_line = ax.axhline(0, color = "cyan", linewidth = 1, label = "scan")

	while idx < height:
		# live cutoff estimate
		cutoff_hz = nyq - idx * freq_bin_width

		if show:
			ax.set_title(f"{cutoff_hz:.2f} Hz")
			scan_line.set_ydata([idx] * len(scan_line.get_xdata()))
			fig.canvas.draw()
			fig.canvas.flush_events()
			plt.pause(0.005) # + time_wait / 2)

		_save_fig(plt, dst, frame_count)
		frame_count += 1

		val = vertical_profile_flipped[idx]
		if val > eps:
			# refine backtrack
			back_step = p / 2
			refined_idx = idx

			while back_step >= 1:
				candidate = int(round(refined_idx - back_step))

				if candidate < 0:
					break

				if vertical_profile_flipped[candidate] > eps:
					refined_idx = candidate

				back_step /= 2

			found_idx = int(round(refined_idx))
			if show:
				scan_line.set_ydata([found_idx] * len(scan_line.get_xdata()))
				scan_line.set_color("lime")
				scan_line.set_linewidth(2)

			cutoff_hz = nyq - found_idx * freq_bin_width

			if show:
				ax.set_title(f"{cutoff_hz:.2f} Hz")

				fig.canvas.draw()
				fig.canvas.flush_events()

			plt.pause(0.01)

			_save_fig(plt, dst, frame_count)
			frame_count += 1

			break

		idx += int(round(p))

	if show:
		plt.ioff()
		plt.show(block = False)
		plt.pause(time_wait)
		plt.close(fig)

	if found_idx is None:
		return None

	nyq = sr / 2
	freq_bin_width = nyq / (height - 1)
	cutoff_hz = nyq - found_idx * freq_bin_width

	return cutoff_hz

def resample_signal(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
	"""
	Resample signal to target sample rate.
	Uses PyDub if available, otherwise falls back to scipy.signal.resample_poly.

	Args:
		y (np.ndarray): Array (mono or channel-first multi-channel).
		orig_sr (int): Original sample rate.
		target_sr (int): Target sample rate.

	Returns:
		np.ndarray: Resampled signal array with same dtype/range as input.
	"""
	if orig_sr <= 0 or target_sr <= 0:
		raise ValueError("Sample rates must be positive integers.")

	# to restore later
	original_dtype = y.dtype

	try:
		from pydub import AudioSegment
		AVAILABLE_PYDUB = True
	except ImportError:
		AVAILABLE_PYDUB = False

	if AVAILABLE_PYDUB:
		import io

		# ensure int16
		if y.dtype != np.int16:
			if np.issubdtype(y.dtype, np.floating):
				y_int16 = np.int16(np.clip(y, -1.0, 1.0) * 32767)
			else:
				y_int16 = y.astype(np.int16)
		else:
			y_int16 = y

		# prep
		if y_int16.ndim == 1:
			raw_audio = y_int16.tobytes()
			channels_for_pydub = 1
		else:
			# (samples, channels)
			y_int16 = y_int16.T
			raw_audio = y_int16.tobytes()
			channels_for_pydub = y_int16.shape[1]

		audio_segment = AudioSegment(
			data = raw_audio,
			sample_width = 2,
			frame_rate = orig_sr,
			channels = channels_for_pydub
		)

		# --- Export through FFmpeg with aresample filter ---
		buf = io.BytesIO()
		audio_segment.export(
			buf,
			format = "wav",
			parameters=["-af", f"aresample={target_sr}:cutoff=1"]
		)

		buf.seek(0)

		# re-read into AudioSegment
		resampled_seg = AudioSegment.from_file(buf, format = "wav")
		samples = np.array(resampled_seg.get_array_of_samples())

		if channels_for_pydub > 1:
			# back to channel-first
			samples = samples.reshape(-1, channels_for_pydub).T

		# restore original dtype range
		if np.issubdtype(original_dtype, np.floating):
			samples = samples.astype(np.float32) / 32767
			samples = samples.astype(original_dtype)
		else:
			samples = samples.astype(original_dtype)

		return samples

	from scipy.signal import resample_poly

	factor = math.gcd(orig_sr, target_sr)
	up = target_sr // factor
	down = orig_sr // factor

	if y.ndim == 2:
		return np.vstack([resample_poly(y[ch], up, down) for ch in range(y.shape[0])]).astype(original_dtype)
	else:
		return resample_poly(y, up, down).astype(original_dtype)