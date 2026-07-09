"""Unit tests for this module

Run with:
	pytest unit.py --cov=axiom --cov-report=term-missing

All audio fixtures are white noise, generated on the fly with a fixed RNG
seed for reproducibility.

Estimated quantities (samplerate, bit depth, peak, cutoff heuristics)
are asserted with inequality.

Deterministic values (channel counts from known signals, method aliases, raised exceptions)
are still asserted with equality.

--------------------------------------------------------------------------

AI/checkpoint-based sample rate estimation (`checkpoint_path = ...`) is
opt-in: it's skipped by default since it needs a real trained checkpoint.

Set the AXIOM_TEST_CHECKPOINT environment variable to a checkpoint path to run it:

	AXIOM_TEST_CHECKPOINT = /path/to/model.pt

[pytest]
markers = 
	ai: requires a real trained checkpoint (set above env var)
)
"""
import os

import numpy as np
import pytest

from axiom import Axiom
from axiom._core.setup import DEPTHS_BIT, SAMPLE_MIN_EXTEND
from axiom._core.algorithms import Estimators, transform_contrast

try:
	import pydub # noqa: F401
	PYDUB_AVAILABLE = True
	del pydub
except ImportError:
	PYDUB_AVAILABLE = False

CHECKPOINT_PATH = os.environ.get("AXIOM_TEST_CHECKPOINT")

# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #

def white_noise(n_samples, channels = 1, seed = 0):
	"""Reproducible white noise in [-1, 1)."""
	n_samples = int(n_samples)
	rng = np.random.default_rng(seed)
	if channels == 1:
		return rng.uniform(-1, 1, size = n_samples).astype(np.float64)
	return rng.uniform(-1, 1, size = (n_samples, channels)).astype(np.float64)

def first_result(result_dict):
	"""
	Result dicts are keyed by whatever `normalize_path` returns,
	which may not equal the original fixture Path (resolved differently,
	case-folded, etc. - especially on Windows). Since these tests only
	ever process one file at a time, just grab that one value.
	"""
	return next(iter(result_dict.values()))

@pytest.fixture
def wav_factory(tmp_path):
	"""
	Returns a callable that writes a white-noise .wav file under tmp_path
	and returns its Path. Usage:
		f = wav_factory("noise.wav", seconds = 1.0, sr = 44100, channels = 2)
	"""
	import soundfile as sf

	def _make(name, seconds = 1.0, sr = 44100, channels = 1, subtype = "PCM_16", seed = 0):
		path = tmp_path / name
		n_frames = int(seconds * sr)
		data = white_noise(n_frames, channels = channels, seed = seed).astype(np.float32)
		sf.write(str(path), data, sr, subtype = subtype)
		return path

	return _make

@pytest.fixture
def empty_wav_factory(tmp_path):
	import soundfile as sf

	def _make(name = "empty.wav", sr = 44100):
		path = tmp_path / name
		sf.write(str(path), np.array([], dtype = np.float32), sr)
		return path

	return _make

# ------------------------------------------------------------------ #
# __init__ / file discovery
# ------------------------------------------------------------------ #

class TestAxiomInit:
	def test_single_file_is_registered(self, wav_factory):
		f = wav_factory("a.wav")
		ax = Axiom([f])
		assert len(ax.files) == 1

	def test_string_path_is_wrapped_in_list(self, wav_factory):
		f = wav_factory("a.wav")
		ax = Axiom(str(f))
		assert len(ax.files) == 1

	def test_invalid_path_raises_value_error(self, tmp_path):
		with pytest.raises(ValueError):
			Axiom([tmp_path / "does_not_exist.wav"])

	def test_directory_non_recursive_excludes_nested_files(self, wav_factory, tmp_path):
		wav_factory("top.wav")
		(tmp_path / "nested").mkdir()
		wav_factory("nested/deep.wav")

		ax = Axiom([tmp_path], recursive = False)
		names = {os.path.basename(str(p)) for p in ax.files}
		assert "top.wav" in names
		assert "deep.wav" not in names

	def test_directory_recursive_includes_nested_files(self, wav_factory, tmp_path):
		wav_factory("top.wav")
		(tmp_path / "nested").mkdir()
		wav_factory("nested/deep.wav")

		ax = Axiom([tmp_path], recursive = True)
		names = {os.path.basename(str(p)) for p in ax.files}
		assert "top.wav" in names
		assert "deep.wav" in names

	def test_non_audio_extensions_are_filtered_out(self, wav_factory, tmp_path):
		wav_factory("keep.wav")
		(tmp_path / "notes.txt").write_text("not audio")

		ax = Axiom([tmp_path])
		names = {os.path.basename(str(p)) for p in ax.files}
		assert "keep.wav" in names
		assert "notes.txt" not in names

	def test_no_valid_files_raises_value_error(self, tmp_path):
		(tmp_path / "notes.txt").write_text("not audio")
		with pytest.raises(ValueError):
			Axiom([tmp_path])

	def test_duplicate_paths_are_deduplicated(self, wav_factory):
		f = wav_factory("a.wav")
		ax = Axiom([f, f, f])
		assert len(ax.files) == 1

	def test_files_are_sorted(self, wav_factory, tmp_path):
		wav_factory("b.wav")
		wav_factory("a.wav")
		wav_factory("c.wav")

		ax = Axiom([tmp_path])
		names = [os.path.basename(str(p)) for p in ax.files]
		assert names == sorted(names)
		assert set(names) == {"a.wav", "b.wav", "c.wav"}

# ------------------------------------------------------------------ #
# method aliases
# ------------------------------------------------------------------ #

class TestAliases:
	@pytest.mark.parametrize("alias_name", ["sr", "samplerate", "samplingrate", "sampling_rate"])
	def test_sample_rate_aliases(self, alias_name):
		assert getattr(Axiom, alias_name) is Axiom.sample_rate

	@pytest.mark.parametrize("alias_name", ["bd", "bit", "bits"])
	def test_bit_depth_aliases(self, alias_name):
		assert getattr(Axiom, alias_name) is Axiom.bit_depth

	@pytest.mark.parametrize("alias_name", ["ndim", "chan", "chans"])
	def test_channels_aliases(self, alias_name):
		assert getattr(Axiom, alias_name) is Axiom.channels

	@pytest.mark.parametrize("alias_name", ["aio", "estim"])
	def test_estimate_aliases(self, alias_name):
		assert getattr(Axiom, alias_name) is Axiom.estimate

# ------------------------------------------------------------------ #
# Estimators.sample_rate / heuristic_cutoff (needs librosa)
# ------------------------------------------------------------------ #

class TestEstimatorsSampleRate:
	@pytest.mark.parametrize("sr", [8000, 22050, 44100, 48000])
	def test_estimated_samplerate_is_positive_and_bounded(self, sr):
		pytest.importorskip("librosa")
		sig = white_noise(sr) # 1 second of white noise at this rate
		estimated = Estimators.sample_rate(sig, sr, checkpoint_path = None, rounded = False)

		# estimation, not an exact reproduction -> range checks, not == 
		assert estimated > 0
		assert estimated >= sr * 0.5
		assert estimated <= sr * 1.5

	def test_multichannel_signal_is_averaged_down_to_mono(self):
		pytest.importorskip("librosa")
		stereo = white_noise(44100, channels = 2)
		# 2D input for `signal.ndim == 2` averaging path -> should not crash
		estimated = Estimators.sample_rate(stereo.T, 44100, checkpoint_path = None)
		assert estimated > 0

	def test_explicit_n_fft_does_not_crash(self):
		pytest.importorskip("librosa")
		sig = white_noise(44100)
		estimated = Estimators.sample_rate(sig, 44100, checkpoint_path = None, n_fft = 1024)
		assert estimated > 0

	@pytest.mark.ai
	@pytest.mark.skipif(not CHECKPOINT_PATH, reason = "set AXIOM_TEST_CHECKPOINT to a trained model to run this")
	def test_sample_rate_with_real_checkpoint(self):
		pytest.importorskip("librosa")
		sig = white_noise(44100)
		estimated = Estimators.sample_rate(sig, 44100, checkpoint_path = CHECKPOINT_PATH)
		assert estimated > 0

# ------------------------------------------------------------------ #
# Estimators.bit_depth
# ------------------------------------------------------------------ #

class TestEstimatorsBitDepth:
	def test_bit_depth_is_within_scanned_range(self):
		sig = white_noise(SAMPLE_MIN_EXTEND)
		result = Estimators.bit_depth(sig, min_depth = 8, max_depth = 32)

		# estimate -> bounds, not exact equality
		assert result >= 8
		assert result <= 32

	def test_bit_depth_return_details_has_expected_keys(self):
		sig = white_noise(SAMPLE_MIN_EXTEND)
		result, details = Estimators.bit_depth(sig, return_details = True)

		assert result >= 8
		for key in ("best_snr", "snr_error", "lsb_score", "all_results"):
			assert key in details

	def test_bit_depth_multichannel_is_averaged(self):
		stereo = white_noise(SAMPLE_MIN_EXTEND, channels = 2)
		result = Estimators.bit_depth(stereo)
		assert result >= 8
		assert result <= 32

	def test_snapped_bit_depth_lands_on_nearest_candidate(self):
		sig = white_noise(SAMPLE_MIN_EXTEND)
		estimated = Estimators.bit_depth(sig)
		nearest = min(DEPTHS_BIT, key = lambda d: abs(d - estimated))
		# not part of Estimators itself, but exercises the same snap()
		# helper Axiom._process_files uses downstream
		from axiom._core.helpers.iterables import snap
		assert snap(estimated, DEPTHS_BIT) == nearest

# ------------------------------------------------------------------ #
# Estimators.channels
# ------------------------------------------------------------------ #

class TestEstimatorsChannels:
	def test_mono_signal_returns_one(self):
		sig = white_noise(20000, channels = 1)
		assert Estimators.channels(sig) == 1

	def test_more_than_two_channels_returns_shape_directly(self):
		multi = white_noise(20000, channels = 4)
		assert Estimators.channels(multi) == 4

	def test_stereo_fallback_without_pydub_raises_nameerror(self):
		"""Documents bug #1 (_channels_chunking called as a bare name)."""
		if PYDUB_AVAILABLE:
			pytest.skip("pydub is installed; the buggy fallback path isn't reached")

		stereo = white_noise(20000, channels = 2)
		with pytest.raises(NameError, match = "_channels_chunking"):
			Estimators.channels(stereo)

	@pytest.mark.skipif(not PYDUB_AVAILABLE, reason = "requires pydub for phase-cancellation detection")
	def test_stereo_independent_channels_detected_as_stereo(self):
		stereo = white_noise(44100, channels = 2) # independent L/R noise
		assert Estimators.channels(stereo) == 2

	@pytest.mark.skipif(not PYDUB_AVAILABLE, reason = "requires pydub for phase-cancellation detection")
	def test_duplicated_mono_channel_detected_as_mono(self):
		mono = white_noise(44100, channels = 1)
		duplicated = np.stack([mono, mono], axis = 1)
		assert Estimators.channels(duplicated) == 1

# ------------------------------------------------------------------ #
# Estimators.bit_rate
# ------------------------------------------------------------------ #

class TestEstimatorsBitRate:
	def test_wav_bitrate_is_exact_product(self):
		# WAV is uncompressed -> deterministic, exact equality is fine here
		result = Estimators.bit_rate("song.wav", sr = 44100, n_channels = 2, bit_depth = 16)
		assert result == 44100 * 2 * 16

	def test_mp3_bitrate_is_clamped_to_valid_range(self):
		from axiom._core.setup import RATES_MP3_BIT

		# absurdly high raw product should clamp down to the max valid MP3 rate
		result = Estimators.bit_rate("song.mp3", sr = 999_999_999, n_channels = 2, bit_depth = 32)
		assert result == max(RATES_MP3_BIT)

		# absurdly low raw product should clamp up to the min valid MP3 rate
		result_low = Estimators.bit_rate("song.mp3", sr = 1, n_channels = 1, bit_depth = 1)
		assert result_low == min(RATES_MP3_BIT)

	def test_ogg_bitrate_is_clamped_to_valid_range(self):
		from axiom._core.setup import RATES_OGG_BIT

		result = Estimators.bit_rate("song.ogg", sr = 999_999_999, n_channels = 2, bit_depth = 32)
		assert result == max(RATES_OGG_BIT)

		result_low = Estimators.bit_rate("song.ogg", sr = 1, n_channels = 1, bit_depth = 1)
		assert result_low == min(RATES_OGG_BIT)

# ------------------------------------------------------------------ #
# Estimators.peak
# ------------------------------------------------------------------ #

class TestEstimatorsPeak:
	def test_peak_linear_is_within_unit_range(self):
		sig = white_noise(20000)
		result = Estimators.peak(sig, unit = "linear")
		assert result > 0
		assert result <= 1.0

	def test_peak_db_is_negative_or_zero(self):
		sig = white_noise(20000)
		result = Estimators.peak(sig, unit = "db")
		assert result <= 0.0

	def test_peak_of_silence_in_db_is_none_or_a_very_low_floor(self):
		silent = np.zeros(20000)
		result = Estimators.peak(silent, unit = "db")
		assert result is None or result <= -100.0

	def test_peak_of_silence_is_zero_in_linear(self):
		silent = np.zeros(20000)
		assert Estimators.peak(silent, unit = "linear") == 0.0

	def test_invalid_unit_raises_value_error(self):
		sig = white_noise(20000)
		with pytest.raises(ValueError):
			Estimators.peak(sig, unit = "bogus")

# ------------------------------------------------------------------ #
# transform_contrast (standalone helper)
# ------------------------------------------------------------------ #

class TestTransformContrast:
	def test_output_stays_within_unit_range(self):
		img = np.linspace(0, 1, 100).reshape(10, 10)
		out = transform_contrast(img, 64)
		assert out.min() >= 0.0
		assert out.max() <= 1.0

	def test_zero_contrast_is_near_identity(self):
		img = np.linspace(0, 1, 100).reshape(10, 10)
		out = transform_contrast(img, 0)
		assert np.allclose(out, img, atol = 1e-6)

	def test_contrast_value_is_clamped_to_valid_range(self):
		img = np.linspace(0, 1, 100).reshape(10, 10)
		# way outside [-127, 127] should behave the same as the clamped edge
		out_over = transform_contrast(img, 100000)
		out_edge = transform_contrast(img, 127)
		assert np.allclose(out_over, out_edge)

# ------------------------------------------------------------------ #
# End-to-end (full _process_files pipeline)
# ------------------------------------------------------------------ #

class TestEndToEnd:
	def test_channels_end_to_end(self, wav_factory):
		f = wav_factory("mono.wav", channels = 1)
		ax = Axiom([f])
		result = first_result(ax.channels())
		assert result["channels"] == 1

	def test_bit_depth_end_to_end(self, wav_factory):
		f = wav_factory("noise.wav", subtype = "PCM_16")
		ax = Axiom([f])
		r = first_result(ax.bit_depth())

		assert "bit depth" in r
		assert "bit depth snapped" in r
		assert r["bit depth"] >= 8
		assert r["bit depth snapped"] in DEPTHS_BIT

	def test_empty_file_raises_value_error(self, empty_wav_factory):
		f = empty_wav_factory()
		ax = Axiom([f])
		with pytest.raises(ValueError):
			ax.bit_depth()

	def test_estimate_end_to_end(self, wav_factory):
		pytest.importorskip("librosa")
		f = wav_factory("noise.wav", channels = 1, sr = 44100)
		ax = Axiom([f])
		r = first_result(ax.estimate())

		for key in ("samplerate", "cutoff", "bit depth", "bit depth snapped", "channels", "bitrate", "peak"):
			assert key in r
		assert r["samplerate"] > 0
		assert r["cutoff"] == r["samplerate"] / 2

	def test_sample_rate_end_to_end(self, wav_factory):
		pytest.importorskip("librosa")
		f = wav_factory("noise.wav", seconds = 1.0, sr = 44100)
		ax = Axiom([f])
		r = first_result(ax.sample_rate())
		assert r["samplerate"] > 0