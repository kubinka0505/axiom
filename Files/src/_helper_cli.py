import tempfile
from datetime import timedelta
from colorama import Fore, Style

try:
	from pydub import AudioSegment
	AVAILABLE_PYDUB = True
except ImportError as e:
	AVAILABLE_PYDUB = False

import numpy as np
import soundfile as sf

from ._setup import *
from ._helper import *
from ._metadata import *
from ._main import resample_signal, Estimators

#-=-=-=-#
# Main

def process_file(
	index: int,
	file: str,
	total: int,

	args
):
	"""
	Processes an audio file by analyzing, trimming, estimating, and potentially resampling it.

	This function:
		1. Loads an audio file
		2. Extracts metadata
		3. Applies chunking and optional trimming, 
		4. Normalizes the audio
		5. Estimates sample rate
		6. Estimates channel count
		7. Estimates bitrate
		8. Estimates peak level
		9. Writes a modified version of the file to disk if certain conditions are met.

	Args:
		index (int): The index of the current file in the batch.
		file (str): Path to the audio file to be processed.
		total (int): Total number of files in the batch.
		args (Namespace): Parsed command-line arguments or a similar configuration object.

	Raises:
		ValueError: If the computed chunk range is invalid (e.g., start >= end).
	"""
	logger.info("Loaded file:\t\t\t{file}".format(
		file = file,
	))

	# Load file
	with sf.SoundFile(file) as f:
		y = f.read(dtype = "float32").T
		sr = f.samplerate

		subtype_string = f.subtype

		n_frames = f.frames
		n_channels = f.channels

		bit_depth = 32 # FLOAT
		if "_" in subtype_string:
			subtype_splitted = subtype_string.split("_")

			if len(subtype_splitted):
				if len(subtype_splitted) > 1:
					bit_depth = 16 # possibly for MP3
				else:
					bit_depth = subtype_splitted[-1] # invalid on FLAC, as 24-bit do exist

		bit_depth = int(bit_depth)
		bitrate = sr * bit_depth * n_channels

	if y.size == 0:
		logger.error(f"{f} is an empty file")
		raise SystemExit(1)

	logger.info("Information:\t\t\t{color_sr}{sr} Hz{reset}, {samples} samples ({color_meta}{time}{reset}), {color_channels}{channels}{reset}, {color_size}{size}{reset}".format(
		sr = sr,
		samples = n_frames,
		time = str(timedelta(seconds = n_frames / sr))[2:-3],
		channels = "Mono" if n_channels < 2 else "Stereo",
		size = file_size(file),

		color_sr = Fore.BLUE,
		color_channels = Fore.LIME if n_channels < 2 else Fore.ORANGE,
		color_size = Fore.PINK,
		color_meta = Fore.GRAY,

		reset = Fore.RESET
	))

	logger.info("-" * 10)

	# Chunk trimming and validation
	start = to_samples(args.start or 0, sr)
	duration = min(to_samples(args.duration or -1, sr), MAX_DURATION)

	if not duration:
		logger.warning("{color_all}Invalid duration ({color_value}{duration}{color_all}), reverting to file length.{reset}".format(
			duration = args.duration,

			color_all = Fore.GOLD,
			color_value = Fore.RED,

			reset = Fore.RESET
		))
		duration = n_frames

	if start < 0:
		start += n_frames

	if duration < 0:
		duration = n_frames - start

	end = min(start + duration, n_frames)

	if start >= end:
		start = 0
		end = min(start + duration, n_frames)

	if n_channels == 1:
		y_trimmed = y[start:end:args.skip_each]
	else:
		y_trimmed = y[:, start:end:args.skip_each]

	# --- Sample processing ---

	logger.debug("Normalizing signal")
	y_trimmed_max = y_trimmed / (np.max(np.abs(y_trimmed)) + 1E-12)

	# for validation
	s_extended = int(float(min(SAMPLE_MIN_EXTEND, SAMPLE_LARGE)))

	if end < s_extended:
		logger.info(f"Extending signal:\t\t{y_trimmed.shape[-1]} -> {s_extended} samples")
		y_trimmed_max = extend_signal(y_trimmed_max, s_extended)

	end = y_trimmed_max.shape[-1]

	if args.verbosity > 0:
		samples_time = str(timedelta(seconds = (end - start) / sr))[2:]

		logger.info("Processing samples:\t\t{start} -> {end} ({color_meta}each {skip}{reset}) ({color_meta}{ftime}{reset}) ({color_meta}{percentage}%{reset})".format(
			start = start,
			end = end,

			skip = args.skip_each,
			ftime = samples_time,
			percentage = round(percentage(end, n_frames), 2),

			color_meta = Fore.GRAY,
			reset = Fore.RESET
		))

		if end >= SAMPLE_LARGE:
			logger.warning("{color_all}Large amount of samples to process. Consider modifying {color_arg}--duration{color_all}.{reset}".format(
				color_all = Fore.GOLD,
				color_arg = Fore.ORANGE,
				reset = Fore.RESET
			))

	# n_clip = 10
	# logger.debug(f"Clipping signal by {n_clip} times")
	# y_trimmed_max = np.clip(y_trimmed * n_clip, -1.0, 1.0)

	#-=-=-=-#
	# Estimations

	# samplerate
	if args.exclude_sample_rate:
		estimated_sr = sr
	else:
		msg = "Estimating sample rate"
		msg += "\t" * 2

		if args.sr_n_fft:
			msg += f"FFT {args.sr_n_fft}"
		elif args.model:
			msg += f"With {args.model}"

		msg = msg.strip()
		logger.debug(msg)

		estimated_sr = Estimators.sample_rate(
			y_trimmed_max,
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

	# bit depth
	if args.exclude_bit_depth:
		estimated_bit_depth = bit_depth
	else:
		logger.debug("Estimating bit depth")
		estimated_bit_depth = Estimators.bit_depth(y_trimmed)

	estimated_bit_depth_snapped = snap_next(estimated_bit_depth, DEPTHS_BIT)

	# channels
	if args.exclude_channels:
		estimated_channels = n_channels
	else:
		logger.debug("Estimating channels")
		estimated_channels = Estimators.channels(y_trimmed)

	# bitrate
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

	logger.debug("Estimating peak")
	estimated_peak = Estimators.peak(y_trimmed, "dB", 3) if not args.exclude_peak else ""

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

	# verbosity 0
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

	swo = should_write_output(index, args)

	if args.force:
		swo = True
	elif all((
			sr == estimated_sr,
			n_channels == estimated_channels,
			estimated_peak == args.normalize
		)):
		swo = False

	if swo:
		logger.debug("Attempting to write file")
		output_path = get_output_path(file, args)

		resampled = y

		if diff_sr:
			resampled = resample_signal(y, sr, estimated_sr)

		if estimated_channels != n_channels:
			resampled = convert_channels(resampled, estimated_channels, args)

		if estimated_peak != args.normalize:
			normalize = args.normalize
		else:
			normalize = False

		output_path = write_file(
			y = resampled,
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
			#raise Exception("An error occured.")
			return

		orig_size = os.path.getsize(file)
		out_size = os.path.getsize(output_path)

		diff_bytes = out_size - orig_size
		difference_size = file_size(abs(diff_bytes))

		percentage_size = round((diff_bytes / orig_size) * 100, 3)
		percentage_operator = "+" if percentage_size > 0 else "-"

		logger.info(
			"Modified file saved to:\t\t{color_value}{out_path}{reset} ({color_size}{size}{reset}) ({color_meta}{diff_op}{difference_size}{reset}) ({color_meta}{diff_op}{perc_size}%{reset})".format(
				out_path = output_path,
				size = file_size(output_path),

				diff_op = percentage_operator,
				difference_size=difference_size,
				perc_size = abs(percentage_size),

				color_value = "" if percentage_operator == "-" else Fore.ORANGE,
				color_size = Fore.PINK,
				color_meta = Fore.GRAY if percentage_operator == "-" else Fore.RED,
				reset = Fore.RESET
			)
		)
	else:
		if args.output:
			logger.warning("Output not written due to not changed parameters. Use --force flag to bypass it.")

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

	Args:
		original_sample_rate (int): The original sample rate of the audio file (in Hz).
		difference (int): The absolute difference between original and estimated sample rates (in Hz).
		n_channels (int): The original number of audio channels.

		estimated_sample_rate (int): The estimated sample rate (in Hz).
		estimated_bit_depth (int): The estimated bit depth.
		estimated_channels (int): The estimated number of audio channels.
		estimated_bitrate (float): The estimated bitrate (in bits per second).
		estimated_peak (float): The estimated peak amplitude in dBFS (decibels relative to full scale).

		args (Namespace): An argparse object.
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

		logger.info("Estimated sample rate:\t\t{color_bg}{color_sr}{sr} Hz{reset} ({color_diff}-{difference} Hz{reset}) ({color_meta}Linear -{percentage}%{reset}) ({color_meta}Perceptual -{percentage_perceptual}%{reset}){reset_all}".format(
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
		))

	# bit depth
	_norm_bd = "{color_meta}Normalized to {color_value}{estimated_bit_depth_snapped} bits{color_meta} for safety{reset}".format(
		estimated_bit_depth_snapped = snap_next(estimated_bit_depth, DEPTHS_BIT),

		color_value = Fore.GOLD,
		color_meta = Fore.GRAY,
		reset = Fore.RESET
	)
	_norm_bd = f"({_norm_bd})"

	if not args.exclude_bit_depth:
		logger.info("Estimated bit depth:\t\t{color_value}{estimated_bit_depth} bits{reset} {_normalize_notice}".format(
			estimated_bit_depth = estimated_bit_depth,
			_normalize_notice = _norm_bd, # if args.output else "",

			color_value = Fore.ORANGE,
			reset = Fore.RESET
		).strip())

	# channels
	estimated_channels_display = "mono"

	if estimated_channels == 2:
		estimated_channels_display = "stereo"
	elif estimated_channels > 2:
		estimated_channels_display = "multichannel"

	estimated_channels_display = estimated_channels_display.title()

	if not args.exclude_channels:
		logger.info("Estimated channels amount:\t{color_channels_estimated}{est_channels}{reset} (Original: {color_channels_original}{orig_channels}{reset})".format(
			est_channels = estimated_channels, # estimated_channels_display
			orig_channels = n_channels,

			color_channels_estimated = Fore.LIME if estimated_channels <= n_channels else Fore.ORANGE,
			color_channels_original = Fore.BLUE if estimated_channels <= n_channels else Fore.RED,

			reset = Fore.RESET
		))

	# bitrate
	if args._calculate_bitrate:
		logger.info("Calculated bitrate:\t\t{color_value}{bitrate} kb/s{reset}".format(
			bitrate = round(estimated_bitrate / 1E3, 3),

			color_value = Fore.MAGENTA,

			reset = Fore.RESET
		))

	# peak
	if not args.exclude_peak:
		operator = "+" if estimated_peak > 0 else "-"
		estimated_peak = abs(estimated_peak)

		color = Fore.RED

		if operator != "+":
			if estimated_peak < 0.1:
				color = Fore.LIME
			elif estimated_peak <= 6.0:
				color = Fore.ORANGE

		logger.info("Estimated peak:\t\t\t{color}{operator}{value} dBFS{reset}".format(
			operator = operator,
			value = estimated_peak,

			color = color,

			reset = Fore.RESET
		))

#-=-=-=-#
# Processing

def convert_channels(y: np.ndarray, channels: int, args):
	"""
	Convert the audio to the estimated number of channels.

	Args:
		y (np.ndarray): Audio array (mono or stereo).
		channels (int): Desired number of channels.
		args (Namespace): Parsed command-line arguments.

	Returns:
		np.ndarray: Converted audio with the desired channel count.
	"""
	if args.exclude_channels:
		return y

	current_channels = y.shape[0] if y.ndim == 2 else 1

	if channels == current_channels:
		return y

	if channels == 1 and y.ndim == 2:
		return np.mean(y, axis = 0)
	elif channels == 2 and y.ndim == 1:
		return np.stack([y, y], axis = 0)

	return y

def write_file(
	y: np.ndarray,
	sr: int,

	output_path: str,

	bit_depth: int,
	channels: int,
	bitrate: int,

	copy_tags_file: str, #= None,
	normalize: float, # = None,

	args
):
	"""
	Writes an audio signal to disk, with optional normalization, lossy format adjustment, and metadata copying.

	This function supports saving to various output formats including WAV, MP3, and OGG.
	For MP3/OGG, it automatically adjusts sample rate and bitrate to valid values and uses PyDub if available (for direct encoding), otherwise it falls back to a temporary WAV-based export.

	Metadata tags and cover art can be copied from another audio file via `copy_tags_file`, using mutagen for consistent tag normalization across formats.

	Args:
		y (np.ndarray): Audio signal data. Shape can be (samples,) for mono or (channels, samples) for stereo/multi-channel.
		sr (int): Sample rate of the audio signal (Hz).
		output_path (str): Target path to save the output audio file.

		bit_depth (int): Placeholder for bit depth.
		channels (int): Number of audio channels in the output file.
		bitrate (int): Target bitrate in bits per second (e.g., 192000 for 192 kbps).

		copy_tags_file (str, optional): Path to an existing audio file from which to copy metadata and cover image.
		normalize (float, optional): Target peak level in dBFS to normalize the signal to (e.g., -1.0 dBFS).

	Returns:
		str: Path to the successfully saved audio file.
		None: If incorrect extension.

	Raises:
		Exception: If file writing or metadata embedding fails critically.

	Notes:
		- MP3/OGG output will have sample rate and bitrate snapped to nearest supported values.
		- If PyDub is installed, it will be used for MP3/OGG writing.
			If not or if it fails, the file is first written as WAV and then re-encoded using `soundfile`.
		- Normalization is applied by scaling the waveform based on estimated peak dB.
		- Metadata (including embedded images) is read using `normalize_tags` and written using `apply_tags` functions.

	Example:
		>>> y = np.random.rand(2, 48000)
		>>> write_file(y, sr = 48000, output_path = "out.ogg", channels = 2, subtype_string = 16, bitrate = 160000)
	"""
	output_path = str(output_path)
	output_name = os.path.basename(output_path)

	if normalize is not None:
		peak = Estimators.peak(y, "dB")

		if peak is not None:
			gain_db = normalize - peak
			gain_linear = 10 ** (gain_db / 20)

			y *= gain_linear

	extension = os.path.splitext(output_path)[-1].strip(".").lower()

	# check
	if not extension in EXTENSIONS_VALID:
		if args.verbosity > -1:
			logger.error(f"Invalid output file extension. ({output_path})")

		return None

	if y.ndim == 2:
		y = y.T

	old_sr = sr
	old_bitrate = bitrate
	_fallback = True

	# adjust sr and bitrate for lossy fmts
	if extension in ("mp3", "ogg"):
		if extension == "mp3":
			bit_depth = None

			if sr not in RATES_MP3_SAMPLE:
				sr = snap_next(sr, RATES_MP3_SAMPLE)

			if bitrate not in RATES_MP3_BIT:
				bitrate = snap_next(bitrate, RATES_MP3_BIT)

		elif extension == "ogg":
			bit_depth = None

			if sr not in RATES_OGG_SAMPLE:
				sr = snap_next(sr, RATES_OGG_SAMPLE)

			if bitrate not in RATES_OGG_BIT:
				bitrate = snap_next(bitrate, RATES_OGG_BIT)

		elif extension == "flac":
			bit_depth = snap_next(bit_depth, DEPTHS_BIT_FLAC)

		logger.info("Snapped old bitrate:\t\t{old_bitrate} kb/s -> {new_bitrate} kb/s ({difference} kb/s) ({color_meta}-{difference_percentage}%{reset})".format(
			old_bitrate = int(old_bitrate / 1E3),
			new_bitrate = int(bitrate / 1E3),

			difference = int(bitrate / 1E3) - int(old_bitrate / 1E3),
			difference_percentage = round(abs(percentage(old_bitrate, bitrate)), 3),

			color_meta = Fore.GRAY,
			reset = Fore.RESET
		))

		logger.info("Snapped old samplerate:\t\t{old_sr} Hz -> {new_sr} Hz".format(
			old_sr = old_sr,
			new_sr = sr,

			difference = sr - old_sr
		))

		if AVAILABLE_PYDUB:
			try:
				## convert y to AudioSegment

				# ensure y is int16 for pydub raw audio data
				if y.dtype != np.int16:
					# normalize float to int16 range
					if np.issubdtype(y.dtype, np.floating):
						y_int16 = np.int16(y * 32767)
					else:
						y_int16 = y.astype(np.int16)
				else:
					y_int16 = y

				# pydub expects shape (samples, channels)
				if y_int16.ndim == 1:
					raw_audio = y_int16.tobytes()
					channels_for_pydub = 1
				else:
					y_int16 = y_int16.T
					raw_audio = y_int16.tobytes()
					channels_for_pydub = y_int16.shape[1]

				audio_segment = AudioSegment(
					data = raw_audio,
					sample_width = 2, # 2 bytes for int16
					frame_rate = sr,
					channels = channels_for_pydub
				)

				_params = [
					"-fflags", "+bitexact",
					"-flags:v", "+bitexact",
					"-flags:a", "+bitexact",
					"-map", "0",
					"-map_metadata", "-1",
				]

				if bit_depth:
					_params.extend([f"aresample=osf=s{bit_depth}"])

				as_e = audio_segment.export(
					output_path,
					format = extension,
					bitrate = str(bitrate),
					parameters = _params
				)

				as_e.close()
			except Exception as e:
				logger.warning(f"PyDub failed to write {extension.upper()} file: {e}")
				_fallback = True
			else:
				_fallback = False
		else:
			_fallback = True

	if _fallback:
		with tempfile.NamedTemporaryFile(suffix = ".wav", delete = False) as tmp_wav_file:
			tmp_wav_path = tmp_wav_file.name

		try:
			sf.write(tmp_wav_path, y, samplerate = sr)
			data, rate = sf.read(tmp_wav_path)
			sf.write(output_path, data, samplerate = rate, format = extension.upper())
		finally:
			if os.path.exists(tmp_wav_path):
				os.remove(tmp_wav_path)
	else:
		with sf.SoundFile(
			output_path,
			mode = "w",
			samplerate = sr,
			channels = channels
		) as f:
			f.write(y)

	# apply metadata
	if copy_tags_file:
		try:
			tags, cover = normalize_tags(copy_tags_file)
			apply_tags(output_path, tags, cover)
		except Exception as e:
			logger.warning(f"Failed to apply metadata: {e}")

	return output_path

#-=-=-=-#
# Helpers

def should_write_output(file_index: int, args):
	"""
	Determine whether output should be written for a given audio file.

	Decision is based on --output and estimated sample rate deviation.

	Args:
		file_index (int): Index of the current file.
		args (Namespace): Parsed command-line arguments.

	Returns:
		bool: True if output should be written, False otherwise.
	"""
	# overwrite in-place
	if args.output is None:
		return False

	if args.output == "":
		if not args.exclude_sample_rate:
			return True

		return False

	# output is file & multiple inputs – skip all but the first
	if len(args.inputs) > 1 and not os.path.isdir(args.output):
		if file_index != 0:
			return False

		if not args.exclude_sample_rate:
			return True

		return False

	# write all qualifying files if output is directory
	if os.path.isdir(args.output):
		if not args.exclude_sample_rate:
			return True

		return False

	return True

def get_output_path(file: str, args):
	"""
	Determine the output path for a processed audio file.

	Args:
		file (str): Input file path.
		args (Namespace): Parsed command-line arguments.

	Returns:
		str: Path to write the output audio file.
	"""
	if args.output == "":
		return file
	elif os.path.isdir(args.output):
		return os.path.join(args.output, os.path.basename(file))
	else:
		return args.output