from .setup import (
	EXTENSIONS_VALID,
	MAX_DURATION,
	SAMPLE_MIN_EXTEND,
	SAMPLE_LARGE,
	RATES_MP3_BIT,
	RATES_MP3_SAMPLE,
	RATES_OGG_BIT,
	RATES_OGG_SAMPLE,
	DEPTHS_BIT,
	DEPTHS_BIT_FLAC,
	TOLERANCE_SUBJECTIVE_SOFT,
	TOLERANCE_SUBJECTIVE_HARD
)

from .._core.setup import logger
from .._core.metadata import normalize_tags, apply_tags

from .._core.helpers.iterables import snap
from .._core.helpers.files import file_size
from .._core.helpers.numbers import percentage, to_readable
from .._core.helpers.audio import (
	to_samples,
	perceptual_difference,
	resample_signal,
	extend_signal,
	truncate_signal,
	trim_signal,
	spectral_gate,
	bit_depth_to_subtype,
	optimize_file_audio
)

from .._core.main import Estimators

import os
import re
import tempfile
from datetime import timedelta
from wavemarks import MarkerFile
from colorama import Fore, Back, Style

try:
	from pydub import AudioSegment
	AVAILABLE_PYDUB = True
except ImportError:
	AVAILABLE_PYDUB = False

import numpy as np
import soundfile as sf

#-=-=-=-#
# Main

def process_file(
	idx: int,
	file: str,
	total: int,

	args
):
	"""
	Processes an audio file by analyzing, trimming, estimating, and potentially resampling it.

	Workflow
	--------
		1. Load an audio file
		2. Extract metadata
		3. Apply chunking and optional trimming
		4. Normalize the audio
		5. Estimate sample rate
		6. Estimate channel count
		7. Estimate bitrate
		8. Estimate peak level
		9. Write a modified version of the file to disk (if certain conditions are met)

	Parameters
	----------
		idx (int):
			The index of the current file in the batch.

		file (str):
			Path to the audio file to be processed.

		total (int):
			Total number of files in the batch.

		args (Namespace):
			Parsed command-line arguments or a similar configuration object.

	Raises
	------
		ValueError:
			If the computed chunk range is invalid (e.g., start >= end).
	"""
	logger.info(
		"Progress:",
		f"{idx + 1}/{total} [{percentage(idx + 1, total):.2f}%]"
	)

	logger.info("Loaded file:", file)

	# Load file
	with sf.SoundFile(file) as f:
		sr = f.samplerate

		signal = f.read(dtype = "float32")
		signal = signal.T

		subtype_string = f.subtype

		n_frames = f.frames
		n_channels = f.channels

		bit_depth = 32 # FLOAT

		if "_" in subtype_string:
			subtype_splitted = subtype_string.split("_")

			if len(subtype_splitted) > 1 and subtype_splitted[-1].isdigit():
				bit_depth = subtype_splitted[-1]

		# invalid on FLAC, as 24-bit does exist
		bit_depth = int(bit_depth)

		bitrate = sr * bit_depth * n_channels

	if not signal.size:
		logger.error(f"{f} is an empty file")
		raise SystemExit(1)

	channels_display = "Mono"

	if n_channels > 1:
		channels_display = "Stereo"
	if n_channels > 2:
		channels_display = "Multichannel"

	logger.info(
		"Information:",
		"{color_sr}{sr} Hz{reset}, "
		"{samples} samples "
		"({color_meta}{time}{reset}), "
		"{color_channels}{channels}{reset}, "
		"{color_size}{size}{reset}".format(
			sr = sr,
			samples = n_frames,

			time = str(timedelta(seconds = n_frames / sr))[2:-3],
			channels = channels_display.capitalize(),
			size = file_size(file),

			color_sr = Fore.BLUE,
			color_channels = Fore.LIME if n_channels < 2 else Fore.ORANGE,
			color_size = Fore.PINK,
			color_meta = Fore.GRAY,

			reset = Fore.RESET
		)
	)

	logger.info("-" * 10)

	# --- Sample processing ---

	# signal_trimmed is the real signal that may eventually be written.
	# Do not extend it before sample-rate estimation.
	signal_trimmed, start, _ = trim_signal(
		signal,

		start = to_samples(args.start or 0, sr),
		duration = to_samples(args.duration, sr) if args.duration is not None else -1,

		skip_each = args.skip_each,
		max_duration = MAX_DURATION,
	)

	# Spectral gate
	if args.spectral_gate_cutoff is not None:
		parsed_bands = _parse_bands(args.spectral_gate_bands, max_sr = sr)

		logger.info(
			"Applying spectral gate",
			f"{args.spectral_gate_cutoff:.2f} dB cutoff"
		)

		for band in parsed_bands:
			logger.info("Spectral gate bands",
				str(parsed_bands.index(band) + 1)
				+ " → ".join((str(x) for x in band))
				+ " [Hz]"
			)

		signal_trimmed = spectral_gate(
			signal_trimmed,
			sr,
			cutoff = args.spectral_gate_cutoff,
			bands = parsed_bands
		)

	# Truncate signal
	if args.trim_threshold_start is not None and args.trim_threshold_end is not None:
		logger.debug("Trimming signal")

		signal_trimmed = truncate_signal(
			signal_trimmed,
			threshold_start = args.trim_threshold_start,
			threshold_end = args.trim_threshold_end
		)

	# --- Estimation-only signal ---

	# The extension is ONLY used to give estimators enough material.
	#
	# It must never become part of signal_trimmed because signal_trimmed
	# represents the actual audio that will eventually be resampled/written.
	current_len = signal_trimmed.shape[-1]
	target_len = int(min(SAMPLE_MIN_EXTEND, SAMPLE_LARGE))

	signal_estimation = signal_trimmed

	if current_len < target_len:
		logger.info(
			"Extending signal for estimation:",
			f"{current_len} -> {target_len} samples"
		)

		signal_estimation = extend_signal(
			signal_trimmed,
			target_len
		)

	# Normalize signals
	signal_trimmed_max = signal_trimmed / (
		np.max(np.abs(signal_trimmed)) + 1e-12
	)

	signal_estimation_max = signal_estimation / (
		np.max(np.abs(signal_estimation)) + 1e-12
	)

	final_len = signal_trimmed.shape[-1]

	if args.verbosity > 0:
		samples_time = str(timedelta(seconds = final_len / sr))[2:]

		logger.info(
			"Processing samples:",
			"{start} -> {end} "
			"({color_meta}each {skip}{reset}) "
			"({color_meta}{ftime}{reset}) "
			"({color_meta}{percentage:.2f}%{reset})".format(
				start = start,
				end = final_len,

				skip = args.skip_each,

				ftime = samples_time,
				percentage = percentage(final_len, n_frames),

				color_meta = Fore.GRAY,
				reset = Fore.RESET
			)
		)

		if final_len >= SAMPLE_LARGE:
			logger.warning(
				"Large amount of samples to process",
				"Consider modifying {color_arg}--duration{color_all}.{reset}".format(
					color_all = Fore.GOLD,
					color_arg = Fore.ORANGE,
					reset = Fore.RESET
				)
			)

	# --- Estimations ---

	# Sample-rate

	if args.exclude_sample_rate:
		estimated_sr = sr
	else:
		msg = "Estimating sample rate"

		if args.sr_n_fft:
			desc = f"FFT {args.sr_n_fft}"
		elif args.model:
			desc = f"With {args.model}"

		msg = msg.strip()

		logger.debug(msg, desc)

		# IMPORTANT:
		# Use signal_estimation_max here, NOT signal_trimmed_max.
		#
		# signal_estimation may contain the temporary extension needed
		# by the estimator, while signal_trimmed remains the real output.
		estimated_sr = Estimators.sample_rate(
			signal_estimation_max,
			sr,

			checkpoint_path = args.model,

			n_fft = args.sr_n_fft,

			freq_step = args.frequency_step,

			show_graph = True if args.verbosity > 1 else False,

			rounded = False
		)

	estimated_cutoff = estimated_sr / 2
	estimated_sr = int(estimated_sr)

	diff_sr = abs(sr - estimated_sr)

	# Bit depth
	if args.exclude_bit_depth:
		estimated_bit_depth = bit_depth
	else:
		logger.debug("Estimating bit depth")

		# Bit depth describes the actual selected signal, not the
		# estimation-only extension.
		estimated_bit_depth = Estimators.bit_depth(signal_trimmed_max)

	if args.spectral_gate_cutoff is None:
		estimated_bit_depth_snapped = snap(estimated_bit_depth, DEPTHS_BIT)
	else:
		# change to 32 if needed
		estimated_bit_depth_snapped = 24

	# Channels
	if args.exclude_channels:
		estimated_channels = n_channels
	else:
		logger.debug("Estimating channels")

		# Again, use the real signal rather than the extended estimation copy.
		estimated_channels = Estimators.channels(signal_trimmed_max)

	# Bitrate
	if args._calculate_bitrate:
		logger.debug("Estimating bitrate")

		estimated_bitrate = Estimators.bit_rate(
			file,
			estimated_sr,
			estimated_channels,
			estimated_bit_depth_snapped
		)
	else:
		estimated_bitrate = bitrate

	# Peak
	if args.exclude_peak:
		estimated_peak = ""
	else:
		logger.debug("Estimating peak")
		estimated_peak = Estimators.peak(signal_trimmed_max, "dB", 3)

	#-=-=-=-#

	log_estimates(
		original_sample_rate = sr,
		difference = diff_sr,
		n_channels = n_channels,

		estimated_sample_rate = estimated_sr,
		estimated_bit_depth = estimated_bit_depth,
		estimated_channels = estimated_channels,
		estimated_bitrate = estimated_bitrate,
		estimated_peak = estimated_peak,

		args = args
	)

	# Verbosity 0
	if not args.verbosity:
		val = " ".join(map(str, (
			"-" if args.exclude_sample_rate else estimated_sr,
			"-" if args.exclude_sample_rate else estimated_cutoff,

			"-" if args.exclude_bit_depth else estimated_bit_depth,
			"-" if args.exclude_bit_depth else estimated_bit_depth_snapped,
			"-" if args.exclude_channels else estimated_channels,
			"-" if args.exclude_bit_rate else estimated_bitrate,

			"-" if args.exclude_peak else estimated_peak
		))).strip()

		print(val.strip())

	# Output
	do_write = should_write_output(idx, args)

	output_path = None

	if args.file_output:
		output_path = get_output_path(file, args)

	err_maps = {
		"exist": "File already exists",
		"same": "Parameters same as in input file"
	}
	err_code = ""

	if args.force:
		do_write = True
	else:
		if output_path and os.path.exists(output_path):
			do_write = False
			err_code = "exist"

		if all((
			sr == estimated_sr,
			n_channels == estimated_channels,
			estimated_peak == args.normalize
		)):
			do_write = False
			err_code = "same"

	# Write output
	if do_write:
		logger.debug("Attempting to write file")

		# IMPORTANT:
		# Start from signal_trimmed, NEVER signal_estimation.
		#
		# This prevents the samples introduced by extend_signal() from
		# appearing in the output.
		resampled = signal_trimmed

		if diff_sr:
			resampled = resample_signal(signal_trimmed, sr, estimated_sr)

		if estimated_channels != n_channels:
			resampled = convert_channels(resampled, estimated_channels, args)

		if estimated_peak != args.normalize:
			normalize = args.normalize
		else:
			normalize = False

		output_path = write_file(
			signal = resampled,
			sr = estimated_sr,

			output_path = output_path,

			bit_depth = estimated_bit_depth_snapped,
			channels = estimated_channels,
			bitrate = estimated_bitrate,

			copy_tags_file = None if args.exclude_metadata else file,

			normalize = normalize,

			args = args
		)

		if not output_path:
			return

		orig_size = os.path.getsize(file)
		out_size = os.path.getsize(output_path)

		diff_bytes = out_size - orig_size
		difference_size = file_size(abs(diff_bytes))

		percentage_size = round((diff_bytes / orig_size) * 100, 3)
		percentage_operator = "+" if percentage_size > 0 else "-"

		logger.info(
			"Modified file saved to:",
			"{color_value}{out_path}{reset} "
			"({color_size}{size}{reset}) "
			"({color_meta}{diff_op}{difference_size}{reset}) "
			"({color_meta}{diff_op}{perc_size}%{reset})".format(
				out_path = output_path,

				size = file_size(output_path),

				diff_op = percentage_operator,
				difference_size = difference_size,
				perc_size = abs(percentage_size),

				color_value = "" if percentage_operator == "-" else Fore.ORANGE,
				color_size = Fore.PINK,
				color_meta = Fore.GRAY if percentage_operator == "-" else Fore.RED,

				reset = Fore.RESET
			)
		)
	elif args.file_output:
		logger.warning("Output not written", err_maps[err_code])
		logger.warning("Use --force flag to bypass it.")

	logger.info("-" * 10)

def log_estimates(
	original_sample_rate: int,
	difference: int,
	n_channels: int,

	estimated_sample_rate: int,
	estimated_bit_depth: int,
	estimated_channels: int,
	estimated_bitrate: float,
	estimated_peak: float,

	args
):
	"""
	Logs the estimated audio properties and their differences from the original.

	This function prints diagnostic information about the estimated audio sample rate, 
	bitrate, number of channels, and peak level, along with comparisons to the original values.

	The level of detail depends on the `verbosity` setting in `args`.

	Parameters
	----------
		original_sample_rate (int):
			The original sample rate of the audio file (in Hz).

		difference (int):
			The absolute difference between original and estimated sample rates (in Hz).

		n_channels (int):
			The original number of audio channels.

		---

		estimated_sample_rate (int):
			The estimated sample rate (in Hz).

		estimated_bit_depth (int):
			The estimated bit depth.

		estimated_channels (int):
			The estimated number of audio channels.

		estimated_bitrate (float):
			The estimated bitrate (in bits per second).

		estimated_peak (float):
			The estimated peak amplitude in dBFS (decibels relative to full scale).

		---

		args (Namespace):
			An argument parser namespace object.
	"""
	if args.verbosity <= 0:
		return

	# samplerate
	if not args.exclude_sample_rate:
		percentage_perceptual = perceptual_difference(original_sample_rate, estimated_sample_rate)

		if difference > TOLERANCE_SUBJECTIVE_HARD:
			color = Fore.RED
		elif difference > TOLERANCE_SUBJECTIVE_SOFT:
			color = Fore.GOLD
		else:
			color = Fore.LIME

		if difference:
			color_sr = color
		else:
			color_sr = Fore.GOLDWHITE

		if percentage_perceptual > 5:
			color = Fore.RED
			color_sr = color

		logger.info(
			"Estimated sample rate:",
			"{color_bg}{color_sr}{sr} Hz{reset} ({color_diff}-{difference} Hz{reset}) ({color_meta}Linear -{percentage:.2f}%{reset}) ({color_meta}Perceptual -{percentage_perceptual:.2f}%{reset}){reset_all}".format(
				sr = estimated_sample_rate,
				difference = difference,
				percentage = round(100 - percentage(estimated_sample_rate, original_sample_rate), 3),
				percentage_perceptual = round(percentage_perceptual, 3),

				color_bg = "" if difference else Back.GOLDDARK,
				color_diff = color,
				color_sr = color_sr,
				color_meta = Fore.GRAY,

				reset = Fore.RESET,
				reset_all = Style.RESET_ALL
			)
		)

	# bit depth
	_norm_bd = "{color_meta}Normalized to {color_value}{estimated_bit_depth_snapped} bits{color_meta} for safety{reset}".format(
		estimated_bit_depth_snapped = snap(estimated_bit_depth, DEPTHS_BIT),

		color_value = Fore.GOLD,
		color_meta = Fore.GRAY,
		reset = Fore.RESET
	)
	_norm_bd = f"({_norm_bd})"

	if not args.exclude_bit_depth:
		logger.info(
			"Estimated bit depth:",
			"{color_value}{estimated_bit_depth} bits{reset} {_normalize_notice}".format(
				estimated_bit_depth = estimated_bit_depth,
				_normalize_notice = _norm_bd, # if args.file_output else "",

				color_value = Fore.ORANGE,
				reset = Fore.RESET
			).strip()
		)

	# channels
	estimated_channels_display = "mono"

	if estimated_channels == 2:
		estimated_channels_display = "stereo"
	elif estimated_channels > 2:
		estimated_channels_display = "multichannel"

	estimated_channels_display = estimated_channels_display.title()

	if not args.exclude_channels:
		logger.info(
			"Estimated channels amount:",
			"{color_channels_estimated}{est_channels}{reset} (Original: {color_channels_original}{orig_channels}{reset})".format(
				est_channels = estimated_channels, # estimated_channels_display
				orig_channels = n_channels,

				color_channels_estimated = Fore.LIME if estimated_channels <= n_channels else Fore.ORANGE,
				color_channels_original = Fore.BLUE if estimated_channels <= n_channels else Fore.RED,

				reset = Fore.RESET
			)
		)

	# bitrate
	if args._calculate_bitrate:
		logger.info(
			"Calculated bitrate:",
			"{color_value}{bitrate} kb/s{reset}".format(
				bitrate = round(estimated_bitrate / 1e3, 3),

				color_value = Fore.MAGENTA,

				reset = Fore.RESET
			)
		)

	# peak
	if not args.exclude_peak:
		operator = "+" if estimated_peak > 0 else "-"
		estimated_peak = abs(estimated_peak)

		color = Fore.RED

		if operator != "+":
			# ESTIMATED PEAK THRESHOLDS
			if estimated_peak < 0.1:
				color = Fore.LIME
			elif estimated_peak <= 6.0:
				color = Fore.ORANGE

		logger.info(
			"Estimated peak:",
			"{color}{operator}{value:.2f} dBFS{reset}".format(
				operator = operator,
				value = estimated_peak,

				color = color,

				reset = Fore.RESET
			)
		)

#-=-=-=-#
# Processing

def convert_channels(signal: np.ndarray, channels: int, args) -> np.ndarray:
	"""
	Convert the audio to the estimated number of channels.

	Parameters
	----------
		signal (np.ndarray):
			Audio array (mono or stereo).

		channels (int):
			Desired number of channels.

		args (Namespace):
			Parsed command-line arguments.

	Returns
	-------
		np.ndarray:
			Converted audio with the desired channel count.
	"""
	if args.exclude_channels:
		return signal

	current_channels = signal.shape[0] if signal.ndim == 2 else 1

	if channels == current_channels:
		return signal

	if channels == 1 and signal.ndim == 2:
		return np.mean(signal, axis = 0)
	elif channels == 2 and signal.ndim == 1:
		return np.stack([signal, signal], axis = 0)

	return signal

def write_file(
	signal: np.ndarray,
	sr: int,
	output_path: str,

	bit_depth: int,
	channels: int,
	bitrate: int,

	copy_tags_file: str,
	normalize: float,

	args
) -> str:
	"""
	Writes an audio signal to disk, with optional normalization, lossy format adjustment, and metadata copying.

	This function supports saving to various output formats including WAV, MP3, and OGG.
	For MP3/OGG, it automatically adjusts sample rate and bitrate to valid values and uses PyDub if available (for direct encoding), otherwise it falls back to a temporary WAV-based export.

	Metadata tags and cover art can be copied from another audio file via `copy_tags_file`, using mutagen for consistent tag normalization across formats.

	Parameters
	----------
		signal (np.ndarray):
			Audio signal data.
			Shape can be (samples,) for mono or (channels, samples) for stereo/multi-channel.

		sr (int):
			Sample rate of the audio signal (Hz).

		output_path (str):
			Target path to save the output audio file.

		---

		bit_depth (int):
			Placeholder for bit depth.

		channels (int):
			Number of audio channels in the output file.

		bitrate (int):
			Target bitrate in bits per second (e.g., 192000 for 192 kbps).

		---

		copy_tags_file (str, optional):
			Path to an existing audio file which to copy metadata and cover image from.

		normalize (float, optional):
			Target peak level in dBFS to normalize the signal to (e.g., -1.0 dBFS).

	Returns
	-------
		str:
			Path to the successfully saved audio file.

		None:
			If incorrect extension.

	Raises
	------
		Exception:
			If file writing or metadata embedding fails critically.

	Notes
	-----
		- MP3/OGG output will have sample rate and bitrate snapped to nearest supported values.
		- If PyDub is installed, it will be used for MP3/OGG writing.
			- If not or if it fails, the file is first written as WAV and then re-encoded using `soundfile`.
		- Normalization is applied by scaling the waveform based on estimated peak dB.

	Example
	-------
		>>> signal = np.random.rand(2, 48000)
		>>> write_file(signal, sr = 48000, output_path = "out.ogg", bit_depth = 16, channels = 2, bitrate = 160000)
	"""
	tags, cover = (None, None)

	if copy_tags_file:
		try:
			tags, cover = normalize_tags(copy_tags_file)
		except Exception as e:
			logger.warning(f"Failed to read metadata: {e}")

	output_path = str(output_path)

	extension = os.path.splitext(output_path)[-1].strip(".").lower()

	if extension not in EXTENSIONS_VALID:
		if args.verbosity > -1:
			logger.error(f"Invalid output file extension. ({output_path})")

		return None

	if signal.ndim == 2:
		signal = signal.T

	_fallback = True

	# adjust sr/bitrate
	if extension == "mp3":
		bit_depth = None

		if sr not in RATES_MP3_SAMPLE:
			sr = snap(sr, RATES_MP3_SAMPLE)

		if bitrate not in RATES_MP3_BIT:
			bitrate = snap(bitrate, RATES_MP3_BIT)

	elif extension == "ogg":
		bit_depth = None

		if sr not in RATES_OGG_SAMPLE:
			sr = snap(sr, RATES_OGG_SAMPLE)

		if bitrate not in RATES_OGG_BIT:
			bitrate = snap(bitrate, RATES_OGG_BIT)
	elif extension == "flac" and bit_depth is not None:
		bit_depth = snap(bit_depth, DEPTHS_BIT_FLAC)

	# compute subtype
	subtype = bit_depth_to_subtype(bit_depth, extension)

	# normalize
	if normalize is not None:
		peak = Estimators.peak(signal, "dB")

		if peak is not None:
			gain_linear = 10 ** ((normalize - peak) / 20)
			signal *= gain_linear

	# PyDub path (unchanged but safe)
	if AVAILABLE_PYDUB and extension in ("mp3", "ogg"):
		try:
			if signal.dtype != np.int16:
				if np.issubdtype(signal.dtype, np.floating):
					signal_scaled = np.clip(signal, -1.0, 1.0)
					signal_scaled = (signal_scaled * 32767).astype(np.int16)
				else:
					signal_scaled = signal.astype(np.int16)
			else:
				signal_scaled = signal

			if signal_scaled.ndim == 1:
				raw_audio = signal_scaled.tobytes()
				channels_for_pydub = 1
			else:
				signal_scaled = signal_scaled.T
				raw_audio = signal_scaled.tobytes()
				channels_for_pydub = signal_scaled.shape[1]

			audio_segment = AudioSegment(
				data = raw_audio,
				sample_width = 2,
				frame_rate = sr,
				channels = channels_for_pydub
			)

			audio_segment.export(
				output_path,
				format = extension,
				bitrate = str(bitrate),
				parameters = [
					"-fflags", "+bitexact",
					"-flags:v", "+bitexact",
					"-flags:a", "+bitexact",
					"-map_metadata", "-1",
				]
			)

			_fallback = False
		except Exception as e:
			logger.warning(f"PyDub failed: {e}")
			_fallback = True
	else:
		_fallback = True

	# SOUND FILE PATH
	if _fallback:
		with tempfile.NamedTemporaryFile(suffix = ".wav", delete = False) as tmp:
			tmp_path = tmp.name

		try:
			sf.write(tmp_path, signal, sr, subtype = subtype)
			data, rate = sf.read(tmp_path)

			kwargs = {
				"format": extension.upper(),
				"samplerate": rate
			}

			if subtype is not None:
				kwargs["subtype"] = subtype

			sf.write(output_path, data, **kwargs)
		finally:
			if os.path.exists(tmp_path):
				os.remove(tmp_path)
	else:
		# ONLY pass subtype for PCM formats
		if extension in ("wav", "flac", "aiff") and subtype is not None:
			with sf.SoundFile(
				output_path,
				mode = "w",
				samplerate = sr,
				channels = channels,
				subtype = subtype
			) as f:
				f.write(signal)
		else:
			with sf.SoundFile(
				output_path,
				mode = "w",
				samplerate = sr,
				channels = channels
			) as f:
				f.write(signal)

	if not args.no_optimize:
		logger.info("Attempting to optimize file")
		output_path = optimize_file_audio(output_path)

	# metadata
	if tags is not None:
		try:
			apply_tags(output_path, tags, cover)
		except Exception as e:
			logger.warning(f"Failed metadata: {e}")

	# markers
	if extension == "wav":
		_ = MarkerFile(copy_tags_file).copy(output_path)

	return output_path

#-=-=-=-#
# Helpers

def clear_colored_text_file(path: str) -> str:
	with open(path, "r", encoding = "UTF-8") as file:
		content = file.read()

	lines = [line for line in content.split("\n") if line.strip()]
	lines = [re.compile(r"\x1b\[[0-9;]*m").sub("", line) for line in lines]
	lines = [re.sub(r"\x1b\[(?:0|39)m", "\n", line) for line in lines]
	lines = [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in lines]
	lines = [line.strip() for line in lines]

	# Remove empty lines
	content = "\n".join(lines).strip()

	with open(path, "w", encoding = "UTF-8") as file:
		file.write(content)

	return path

def _parse_bound(v) -> float:
	if isinstance(v, (int, float)):
		if v == -1:
			return float("inf")

		return float(v)

	v = str(to_readable(v)).strip().lower().replace(" ", "")

	if v in ("inf", "+inf", "infinity", "+infinity"):
		return float("inf")

	val = float(v)

	# shorthand
	if val == -1:
		return float("inf")

	return val

def _parse_bands(input_map: str, delim_parts: str = ":", delim_vals: str = ",", max_sr: int = None) -> list:
	if not input_map:
		return None

	input_map = input_map.strip().strip(delim_parts).lower().replace(" ", "")

	bands = list()

	for part in input_map.split(delim_parts):
		parts = part.split(delim_vals)

		low = _parse_bound(parts[0])

		if len(parts) < 2 or not parts[1]:
			high = -1
		else:
			high = _parse_bound(parts[1])

		if max_sr and high == float("inf"):
			high = max_sr

		bands.append((low, high))

	bands = list(set(bands))

	if not bands:
		return None

	return bands

def should_write_output(file_index: int, args) -> bool:
	"""
	Determine whether output should be written for a given audio file.

	Decision is based on output and estimated sample rate deviation.

	Parameters
	----------
		file_index (int):
			Index of the current file.

		args (Namespace):
			Parsed command-line arguments.

	Returns
	-------
		bool:
			True if output should be written, False otherwise.
	"""
	# overwrite in-place
	if args.file_output is None:
		return False

	if args.file_output == "":
		if not args.exclude_sample_rate:
			return True

		return False

	# output is file & multiple inputs – skip all but the first
	if len(args.file_input) > 1 and not os.path.isdir(args.file_output):
		if file_index != 0:
			return False

		if not args.exclude_sample_rate:
			return True

		return False

	# write all qualifying files if output is directory
	if os.path.isdir(args.file_output):
		if not args.exclude_sample_rate:
			return True

		return False

	return True

def get_output_path(file: str, args) -> str:
	"""
	Determine the output path for a processed audio file.

	Parameters
	----------
		file (str):
			Input file path.

		args (Namespace):
			Parsed command-line arguments.

	Returns
	-------
		str:
			Path to write the output audio file.
	"""
	retval = args.file_output

	if args.file_output == "":
		retval = file
	elif os.path.isdir(args.file_output):
		retval = os.path.join(args.file_output, os.path.basename(file))

	return retval