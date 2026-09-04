import os
import re
import math
import subprocess
import numpy as np
import soundfile as sf
from io import BytesIO
from pytimeparse.timeparse import timeparse

from .numbers import to_readable
from .files import package_search

# Properties
def to_samples(value, sr: int) -> int | None:
	"""
	Converts time strings into samples.

	Examples
	--------
		- "2s"
		- "00:01"
		- "500ms"
	"""
	if value is None:
		return None

	s = str(value).strip().lower()

	# raw number = samples
	num = to_readable(s)
	if isinstance(num, (int, float)):
		return int(num)

	m = re.fullmatch(r"([0-9]*\.?[0-9]+(?:e[+-]?\d+)?)(ms|s)?", s)

	if m:
		n, unit = m.groups()
		n = float(n)

		if unit == "ms":
			return int((n / 1000) * sr)
		if unit == "s":
			return int(n * sr)

		# if no unit but wasn't parsed earlier, treat as samples fallback
		return int(n)

	# fallback to timeparser (00:01 etc.)
	parsed = timeparse(s)

	if parsed is not None:
		return int(parsed * sr)

	return None

def to_db(value: float) -> float:
	"""
	Converts magnitude to decibels.

	Parameters
	----------
		value (float):
			value to be converted to decibels.

	Returns
	-------
		float:
			Converted result.
	"""
	return 20 * np.log10(value + 1e-12)

def bit_depth_to_subtype(bit_depth: int, fmt: str) -> str:
	fmt = fmt.lower()

	if fmt == "wav":
		if bit_depth == 16:
			return "PCM_16"

		if bit_depth == 24:
			return "PCM_24"

		if bit_depth == 32:
			return "FLOAT"

		return "PCM_32"

	if fmt == "flac":
		if bit_depth == 16:
			return "PCM_16"

		if bit_depth == 24:
			return "PCM_24"

		return "PCM_24"

	return None

def perceptual_difference(sr: int, est_sr: int, min_freq: float = 20.0, max_freq: float = 20000.0) -> float:
	"""
	Calculates the perceptual difference between the original and estimated sample rates.

	This function estimates how perceptually different two sample rates are
	by comparing their Nyquist frequencies on a logarithmic (pitch-like) scale.

	The result is expressed as a relative percentage difference.

	Parameters
	----------
		sr (int):
			Original sample rate (Hz).

		est_sr (int):
			Estimated sample rate (Hz).

		---

		min_freq (float, optional):
			Minimum frequency to consider for perceptual comparison.

		max_freq (float, optional):
			Maximum frequency to consider for perceptual comparison.

	Returns
	-------
		float:
			Perceptual difference as a percentage. Higher values indicate greater perceptual deviation.

	Example
	-------
		>>> perceptual_difference(44100, 32000)
		2.253181519444287
	"""
	est_freq = max(min(est_sr / 2, max_freq), min_freq)
	orig_freq = max(min(sr / 2, max_freq), min_freq)

	# perceptual scale: log10 of freq
	log_est = np.log10(est_freq)
	log_orig = np.log10(orig_freq)

	# relative perceptual diff %
	return 100 * abs(log_orig - log_est) / log_orig

# Manipulation
def resample_signal(
	signal: np.ndarray,

	orig_sr: int,
	target_sr: int,

	force_scipy: bool = False
) -> np.ndarray:
	"""
	Resample signal to target sample rate.
	Uses PyDub if available, otherwise falls back to scipy.signal.resample_poly.

	Parameters
	----------
		signal (np.ndarray):
			Array (mono or channel-first multi-channel).

		orig_sr (int):
			Original sample rate.

		target_sr (int):
			Target sample rate.

		force_scipy (bool):
			Force resample_poly even when PyDub is available.

	Returns
	-------
		np.ndarray:
			Resampled signal array with same dtype/range as input.
	"""
	if orig_sr <= 0 or target_sr <= 0:
		raise ValueError("Sample rates must be positive integers.")

	# to restore later
	original_dtype = signal.dtype

	if not force_scipy:
		try:
			from pydub import AudioSegment
			AVAILABLE_PYDUB = True
		except ImportError:
			AVAILABLE_PYDUB = False

	if not force_scipy and AVAILABLE_PYDUB:
		# ensure int16
		if signal.dtype != np.int16:
			if np.issubdtype(signal.dtype, np.floating):
				signal_scaled = np.int16(
					np.clip(signal, -1.0, 1.0) * 32767
				)
			else:
				signal_scaled = signal.astype(np.int16)
		else:
			signal_scaled = signal

		# prep
		if signal_scaled.ndim == 1:
			raw_audio = signal_scaled.tobytes()
			channels_for_pydub = 1
		else:
			# (samples, channels)
			signal_scaled = signal_scaled.T
			raw_audio = signal_scaled.tobytes()
			channels_for_pydub = signal_scaled.shape[1]

		audio_segment = AudioSegment(
			data = raw_audio,
			sample_width = 2,
			frame_rate = orig_sr,
			channels = channels_for_pydub
		)

		# export through FFmpeg with aresample filter
		buf = BytesIO()

		audio_segment.export(
			buf,
			format = "wav",
			parameters = ["-af", f"aresample={target_sr}:cutoff=1"]
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

	if signal.ndim == 2:
		return np.vstack([
			resample_poly(signal[ch], up, down)
			for ch in range(signal.shape[0])
		]).astype(original_dtype)
	else:
		return resample_poly(signal, up, down).astype(original_dtype)

def extend_signal(signal: np.ndarray, target_length: int) -> np.ndarray:
	"""
	Extends an ndarray by repeating along the last axis until it reaches at least `target_length`.

	If the repeated signal exceeds the target length, it is trimmed to fit exactly.

	Parameters
	----------
		signal (np.ndarray):
			Input array, last dimension is time axis.

		target_length (int):
			Desired minimum length along last axis.

	Returns
	-------
		np.ndarray:
			Extended array with last axis length exactly `target_length`.

	Example
	-------
		>>> extend_signal(np.array([1, 2, 3]), 7)
		array([1, 2, 3, 1, 2, 3, 1])
		
		>>> extend_signal(np.array([[1,2,3],[4,5,6]]), 7)
		array([[1,2,3,1,2,3,1],
			   [4,5,6,4,5,6,4]])
	"""
	idx = np.arange(target_length) % signal.shape[-1]
	return np.take(signal, idx, axis = -1)

def spectral_gate(
	signal: np.ndarray,
	sr: int,

	cutoff: float = None,
	knee: float = 10.0,

	bands: list[list[float]] | None = None,

	fft_size: int = 2048,
	overlap: float = 0.5,

	window: str = "Hann",

	diff: bool = False,
) -> np.ndarray:
	"""
	Spectral gate / soft-knee frequency-domain noise suppressor.

	Uses STFT / ISTFT processing in the frequency domain and
	applies a soft-knee gain curve based on spectral magnitude in dB.

	Higher overlap improves stability and reduces artifacts.

	Parameters
	----------
		signal (np.ndarray):
			Input audio signal. Can be mono (shape: [n]) or stereo (shape: [n, channels]).
			Expected float32 in range [-1.0, 1.0].

		sr (int):
			Sample rate of the input audio in Hz.

		---

		cutoff (float, optional):
			Absolute dB threshold for attenuation.
			Frequency bins below this level are progressively attenuated.
			Returns input signal if None.

		knee (float, optional):
			Soft transition width around the cutoff threshold in dB.
			Controls how gradually gain transitions from 0 → 1.

		---

		bands (list[list[float]] | list[float] | None, optional):
			Frequency ranges in Hz where spectral gating is active.

			Accepts either a single range [low, high] or multiple ranges
			[[low1, high1], [low2, high2], ...].

			Frequencies outside these ranges bypass the gate.

			If None, all frequencies are processed.

		---

		fft_size (int, optional):
			Size of the FFT window used for STFT analysis.
			Larger values improve frequency resolution but increase latency.

		overlap (float, optional):
			Fraction of FFT window overlap (0.0–0.95 typical).
			Higher values reduce artifacts but increase computation cost.

		window (str, optional):
			Window function applied before FFT (e.g., "hann", "blackman").

		---

		diff (bool, optional):
			Returns original signal spectrally subtracted from the result.

	Returns
	-------
		np.ndarray:
			Processed audio signal (float32), same shape as input, with values clamped to [-1.0, 1.0].
	"""
	if cutoff is None:
		return signal
	else:
		cutoff = -abs(cutoff)

	if bands is not None:
		# shorthand
		if (
			isinstance(bands, (list, tuple))
			and len(bands) == 2
			and all(isinstance(x, (int, float)) for x in bands)
		):
			bands = [bands]

		# Validate
		for band in bands:
			if not (
				isinstance(band, (list, tuple))
				and len(band) == 2
			):
				raise ValueError(
					"bands must be None, [low, high], or a list of [low, high] pairs"
				)

	orig_len = signal.shape[-1]
	hop = int(fft_size * (1 - overlap))

	from scipy.signal import stft, istft

	def process(sig: np.ndarray) -> np.ndarray:
		freqs, _, Zxx = stft(
			sig,
			fs = sr,
			nperseg = fft_size,
			noverlap = fft_size - hop,
			window = window.lower(),
			boundary = "zeros",
			padded = True,
		)

		mag = np.abs(Zxx)
		db = to_db(mag)

		gain = (db - cutoff + knee) / (2.0 * knee)
		gain = np.clip(gain, 0.0, 1.0)

		if bands is not None:
			freq_mask = np.zeros(len(freqs), dtype = bool)

			for low, high in bands:
				freq_mask |= (freqs >= low) & (freqs <= high)

			gain = np.where(freq_mask[:, None], gain, 1.0)

		Zxx *= gain

		_, out = istft(
			Zxx,
			fs = sr,
			nperseg = fft_size,
			noverlap = fft_size - hop,
			window = window.lower(),
		)

		return out.astype(np.float32)

	# mono / stereo
	if signal.ndim == 1:
		signal_result = process(signal)[:orig_len]
	else:
		signal_result = np.stack(
			[process(signal[i]) for i in range(signal.shape[0])],
			axis = 0,
		)
		signal_result = signal_result[:, :orig_len]

	signal_result = np.clip(signal_result, -1.0, 1.0)

	return signal_result.astype(np.float32)

def trim_signal(
	signal: np.ndarray,

	start: int = 0,
	duration: int = -1,
	skip_each: int = 1,
	max_duration: int | None = None,
) -> tuple[np.ndarray, int, int]:
	"""
	Returns
	-------
		tuple:
			(trimmed_signal, start, end)
	"""
	n_frames = signal.shape[-1]

	if start < 0:
		start += n_frames

	start = max(0, min(start, n_frames))

	if duration is None or duration < 0:
		end = n_frames
	else:
		duration = int(duration)

		if max_duration is not None:
			duration = min(duration, max_duration)

		end = start + duration

	end = max(start, min(end, n_frames))

	if end <= start:
		raise ValueError(
			f"Invalid trim range ({start}:{end}) for {n_frames} samples"
		)

	step = max(1, int(skip_each))

	if signal.ndim == 1:
		trimmed = signal[start:end:step]
	else:
		trimmed = signal[..., start:end:step]

	return trimmed, start, end

def truncate_signal(
	signal: np.ndarray,
	threshold_left: float = None,
	threshold_right: float = None,
	threshold_step: float = 5.0,
) -> np.ndarray:
	"""
	Trim silence independently from the left and right side.

	Parameters
	----------
	signal:
		Array (mono or channel-first multi-channel).

	threshold_left:
		dB threshold for the left side.
		None disables left trimming.

	threshold_right:
		dB threshold for the right side.
		None disables right trimming.

	threshold_step:
		Amount by which to relax the threshold if no sample survives.
	"""
	if threshold_left:
		threshold_left = abs(threshold_left)
	else:
		return signal

	if threshold_right:
		threshold_right = abs(threshold_right)
	else:
		return signal

	threshold_step = abs(threshold_step)

	if signal.ndim == 1:
		amplitude = np.abs(signal)
	else:
		amplitude = np.max(np.abs(signal), axis = 0)

	reference = np.max(amplitude)

	if reference <= 0:
		return signal

	db = to_db(
		np.maximum(
			amplitude / reference,
			np.finfo(float).tiny,
		)
	)

	start = 0
	end = signal.shape[-1]

	# right
	if threshold_left is not None:
		current_threshold = -abs(float(threshold_left))

		while True:
			indices = np.flatnonzero(
				(db >= current_threshold) &
				(amplitude > 0)
			)

			if indices.size:
				start = indices[0]
				break

			current_threshold += threshold_step

			if current_threshold >= 0:
				start = 0
				break

	# right
	if threshold_right is not None:
		current_threshold = -abs(float(threshold_right))

		while True:
			indices = np.flatnonzero(
				(db >= current_threshold) &
				(amplitude > 0)
			)

			if indices.size:
				end = indices[-1] + 1
				break

			current_threshold += threshold_step

			if current_threshold >= 0:
				end = signal.shape[-1]
				break

	# forbid empty result
	if start >= end:
		return signal

	return signal[..., start:end]

def optimize_file_audio(file_input: str | BytesIO, level: int = 8) -> str:
	file_input = str(file_input)

	with sf.SoundFile(file_input) as f:
		sr = f.samplerate

	splitext = os.path.splitext(file_input)
	_, ext = splitext
	ext = ext.lower().lstrip(".")

	file_output = "_opt".join(splitext)
	args = []

	if ext == "flac":
		VALID_RATES_SAMPLE_REPLAYGAIN = 44100, 48000

		converter = package_search("flac")

		if not converter:
			raise FileNotFoundError("Encoder not found in PATH")

		args = [converter]

		if level is not None:
			level = max(1, min(level, 8))
			args.append(f"-{level}")

		args += [
			"-e",
			"-p",
			"--verify",
			"-f",
			"--replay-gain" if sr in VALID_RATES_SAMPLE_REPLAYGAIN else None,
			"-s",
			file_input,
			"-o",
			file_output,
		]

		args = [arg for arg in args if arg is not None]

		try:
			subprocess.run(args, check = True)
		except subprocess.CalledProcessError as e:
			print("Conversion failed:", e)
			return file_input

	if args:
		os.remove(file_input)
		os.rename(file_output, file_input)

	return file_input