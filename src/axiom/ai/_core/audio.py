import os
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

#-=-=-=-#

class Processor:
	def __init__(self, signal: np.ndarray):
		self.signal = signal

	def smooth(self) -> np.ndarray:
		self.signal = np.convolve(
			self.signal,
			np.array([0.25, 0.5, 0.25]),
			mode = "same"
		)

		return self.signal

	def lowpass(self, sr: int, cutoff: float, order: int = 16) -> np.ndarray:
		nyq = sr / 2
		cutoff = min(cutoff, nyq)

		sos = butter(
			N = order,
			Wn = cutoff / nyq,
			btype = "low",
			output = "sos"
		)

		self.signal = sosfilt(sos, self.signal)
		return self.signal

	def trim_silence(self, threshold: float) -> np.ndarray:
		energy = np.abs(self.signal)
		above = np.where(energy > threshold)[0]

		if len(above) == 0:
			self.signal = np.array([])
			return self.signal

		start, end = above[0], above[-1]
		self.signal = self.signal[start:end]

		return self.signal

#-=-=-=-#

def process_file(src: str, dst: str, target_sr_effective: int) -> str:
	signal, sr = sf.read(src, dtype = "float32")

	if signal.ndim > 1:
		signal = signal.mean(axis = 1)

	cutoff = target_sr_effective / 2

	p = Processor(signal)

	signal = p.lowpass(sr, cutoff, order = 16)
	signal = p.trim_silence(threshold = 1e-4)

	os.makedirs(os.path.dirname(dst), exist_ok = True)

	sf.write(dst, signal, sr, subtype = "PCM_32")

	return dst