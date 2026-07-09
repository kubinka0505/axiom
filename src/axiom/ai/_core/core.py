import os
import json
import logging
import warnings
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple, List, Union

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category = ResourceWarning) # pid
warnings.filterwarnings("ignore", category = UserWarning, message = "TypedStorage is deprecated")

logger = logging.getLogger(__name__)

class Architectures:
	class ResidualBlock(nn.Module):
		def __init__(self, in_ch, out_ch, stride = (1, 1), dropout = 0.0):
			super().__init__()

			self.conv1 = nn.Conv2d(
				in_ch, out_ch, kernel_size = 3, stride = stride, padding = 1, bias = False
			)
			self.norm1 = nn.GroupNorm(num_groups = min(8, out_ch), num_channels = out_ch)

			self.conv2 = nn.Conv2d(
				out_ch, out_ch, kernel_size = 3, padding = 1, bias = False
			)
			self.norm2 = nn.GroupNorm(num_groups = min(8, out_ch), num_channels = out_ch)

			self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

			if in_ch != out_ch or stride != (1, 1):
				self.skip = nn.Sequential(
					nn.Conv2d(in_ch, out_ch, kernel_size = 1, stride = stride, bias = False),
					nn.GroupNorm(num_groups = min(8, out_ch), num_channels = out_ch),
				)
			else:
				self.skip = nn.Identity()

		def forward(self, x) -> torch.Tensor:
			identity = self.skip(x)

			x = F.gelu(self.norm1(self.conv1(x)))
			x = self.dropout(x)
			x = self.norm2(self.conv2(x))

			return F.gelu(x + identity)

	class CutoffNet(nn.Module):
		"""
		Convolutional residual network for predicting a single cutoff value from
		log-mel spectrogram inputs.

		The model processes spectrograms of shape `(B, 1, F, T)`, where `B` is
		the batch size, `F` is the number of mel frequency bins, and `T` is the
		number of time frames. A convolutional stem is followed by a stack of
		residual blocks that progressively increase channel capacity while reducing
		time and frequency resolution.

		After encoding, features are pooled across the temporal dimension. If
		sequence lengths are provided, masked mean pooling is applied to exclude
		padding; otherwise, a simple mean over time is used. The pooled feature map
		is flattened and passed through a multilayer perceptron head that produces a
		single scalar prediction per sample.

		Input
		-----
			x (Tensor):
				Input spectrogram tensor of shape `(B, 1, n_mels, T)`.

			lengths (Tensor, optional):
				Original sequence lengths (in frames) before padding. Used for
				masked temporal pooling.

		Returns
		-------
			Tensor:
				Shape `(B, 1)` containing raw regression outputs. Any clipping
				or post-processing should be applied outside the model.
		"""
		def __init__(self, n_mels: Optional[int] = 128):
			"""
			Parameters
			----------
				n_mels (int, optional):
					Number of mel frequency bins in the input spectrogram.
					Default is 128.
			"""
			super().__init__()

			self.stem = nn.Sequential(
				nn.Conv2d(1, 32, kernel_size = 5, padding = 2, bias = False),
				nn.GroupNorm(8, 32),
				nn.GELU(),
			)

			self.encoder = nn.Sequential(
				Architectures.ResidualBlock(32, 32, dropout = 0.05),
				Architectures.ResidualBlock(32, 64, stride = (1, 2), dropout = 0.05),
				Architectures.ResidualBlock(64, 96, stride = (2, 2), dropout = 0.1),
				Architectures.ResidualBlock(96, 128, stride = (2, 2), dropout = 0.1),
			)

			reduced_mels = n_mels // 4

			self.head = nn.Sequential(
				nn.Flatten(),
				nn.Linear(128 * reduced_mels, 256),
				nn.GELU(),
				nn.Dropout(0.2),
				nn.Linear(256, 64),
				nn.GELU(),
				nn.Linear(64, 1),
			)

		def forward(self, x, lengths = None) -> torch.Tensor:
			x = self.stem(x)
			x = self.encoder(x)

			if lengths is None:
				# (B, C, F, T) -> average only over T
				x = x.mean(dim = -1)
			else:
				# Time is reduced by 2 * 2 * 2 = 8 through encoder strides.
				reduced_lengths = torch.div(lengths + 7, 8, rounding_mode = "floor")
				reduced_lengths = reduced_lengths.clamp(min = 1, max = x.shape[-1])

				time_index = torch.arange(
					x.shape[-1], device = x.device
				).view(1, 1, 1, -1)

				mask = time_index < reduced_lengths.view(-1, 1, 1, 1)
				mask = mask.to(dtype = x.dtype)

				x = (x * mask).sum(dim = -1) / reduced_lengths.view(-1, 1, 1)

			x = x.flatten(start_dim = 1)

			# return raw linear output, clip happens at inference
			# (see `inference()` below), not during training.
			return self.head(x)

#-=-=-=-#
# Target scaling helpers

def _cutoff_to_target(cutoff_hz: float, nyquist: float, min_hz: float = 20.0) -> float:
	"""
	Map a cutoff frequency to a normalized log-scale target.

	Log scaling spreads out low-frequency cutoffs instead of compressing
	them near 0 the way a plain linear ratio (cutoff / nyquist) does, and
	pairs with an unbounded (no-sigmoid) model output so gradients don't
	vanish near the boundaries of the target range.

	Parameters
	----------
		cutoff_hz (float):
			Cutoff frequency in Hz.

		nyquist (float):
			Nyquist frequency (sr / 2) for this sample.

		min_hz (float):
			Floor to avoid log(0); cutoffs below this are clamped up.

	Returns
	-------
		float:
			Normalized target in [0, 1].
	"""
	cutoff_hz = max(float(cutoff_hz), min_hz)
	log_min = np.log(min_hz)
	log_max = np.log(nyquist)
	log_val = np.log(cutoff_hz)

	target = (log_val - log_min) / (log_max - log_min)
	return float(np.clip(target, 0.0, 1.0))

def _target_to_cutoff(target: float, nyquist: float, min_hz: float = 20.0) -> float:
	"""
	Inverse of _cutoff_to_target — maps a normalized log-scale prediction
	(possibly slightly outside [0, 1] from an unbounded model head) back
	to a cutoff frequency in Hz.

	Parameters
	----------
		target (float):
			Normalized model output.

		nyquist (float):
			Nyquist frequency (sr / 2) for this sample.

		min_hz (float):
			Floor used symmetrically with _cutoff_to_target.

	Returns
	-------
		float:
			Estimated cutoff frequency in Hz, clamped to [min_hz, nyquist].
	"""
	target = float(np.clip(target, 0.0, 1.0))

	log_min = np.log(min_hz)
	log_max = np.log(nyquist)
	log_val = target * (log_max - log_min) + log_min

	cutoff_hz = np.exp(log_val)
	return float(np.clip(cutoff_hz, min_hz, nyquist))

#-=-=-=-#
# Dataset

class CutoffDataset(Dataset):
	def __init__(self, json_map_path: str, audio_dir: str, sample_length: Optional[float] = None):
		"""
		Dataset for audio files mapped to cutoff frequencies.

		Parameters
		----------
			json_map_path (str):
				Path to JSON file mapping filenames to cutoff frequencies.

			audio_dir (str):
				Directory containing audio files.

			sample_length (Optional[float]):
				Fixed length of audio samples in seconds; if None, use full length.
		"""
		with open(json_map_path, "r") as f:
			self.mapping = json.load(f)

		if not self.mapping:
			raise ValueError(
				f"Empty label file: {json_map_path}"
			)

		self.audio_dir = Path(audio_dir)

		total_entries = len(self.mapping)

		self.keys = [
			k for k in self.mapping.keys()
			if (self.audio_dir / k).exists()
		]

		# warn on silently dropped entries instead of just producing
		# a smaller-than-expected dataset with no trace.
		dropped = total_entries - len(self.keys)
		if dropped > 0:
			missing_preview = [
				k for k in self.mapping.keys()
				if not (self.audio_dir / k).exists()
			][:5]
			logger.warning(
				"CutoffDataset: %d/%d label entries skipped (audio file not found in %s). "
				"Examples: %s%s",
				dropped, total_entries, self.audio_dir,
				missing_preview, " ..." if dropped > len(missing_preview) else ""
			)

		if len(self.keys) == 0:
			raise ValueError(
				"No valid audio files found for given labels."
			)

		self.sample_length = sample_length

	def __len__(self):
		"""
		Return number of audio samples.
		"""
		return len(self.keys)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
		"""
		Get the mel spectrogram, target cutoff frequency, and sample rate for an audio sample.

		Parameters
		----------
			idx (int):
				Index of the sample.

		Returns
		-------
			Tuple containing:
				- x (torch.Tensor):
					Mel spectrogram tensor of shape (1, freq, time).

				- y_target (torch.Tensor):
					Target cutoff, normalized on a log-frequency scale (scalar tensor).
					See _cutoff_to_target.

				- sr (int):
					Sample rate of the audio.
		"""
		fname = self.keys[idx]
		cutoff_hz = self.mapping[fname]
		path = os.path.join(self.audio_dir, fname)

		signal, sr = librosa.load(path, sr = None)

		if self.sample_length is not None:
			target_len = int(self.sample_length * sr)

			if len(signal) < target_len:
				signal = np.pad(signal, (0, target_len - len(signal)))
			else:
				signal = signal[:target_len]

		mel = librosa.feature.melspectrogram(y = signal, sr = sr, n_mels = 128, n_fft = 2048, hop_length = 512)
		log_mel = librosa.power_to_db(mel, ref = np.max)

		# per-sample standardization
		log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
		x = torch.tensor(log_mel, dtype = torch.float32).unsqueeze(0) # (1, freq, time)

		nyquist = sr / 2.0

		target = _cutoff_to_target(cutoff_hz, nyquist)

		y_target = torch.tensor([target], dtype = torch.float32)

		return x, y_target, sr

def pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> tuple:
	"""
	Pads lengths in a batch along the time dimension for DataLoader compatibility.

	This function is used as a custom `collate_fn` in a PyTorch DataLoader.
	It takes a list of (input, target, sample_rate) tuples and zero-pads the input tensors along the time dimension so they can be stacked into a single batch tensor.

	Assumes each input `x` has shape (1, freq, time), and all inputs have the same frequency dimension.

	Parameters
	----------
		batch (list of tuples):
			A list of samples, where each sample is a tuple (x, y, sr):

			- x (torch.Tensor):
				Input tensor of shape (1, freq, time).

			- y (torch.Tensor):
				Target tensor (any shape).

			- sr (int):
				Sample rate (passed through unchanged).

	Returns
	-------
		tuple:
			- x_batch (torch.Tensor):
				Batched and padded input tensor of shape (B, 1, freq, max_time).

			- y_batch (torch.Tensor):
				Batched target tensor.

			- srs (tuple):
				Tuple of original sample rates for each sample in the batch.

	Example
	-------
		>>> loader = DataLoader(dataset, batch_size = 8, collate_fn = pad_collate)
	"""
	# batch: list of (x, y, sr)
	xs, ys, srs = zip(*batch)

	lengths = torch.tensor([x.shape[-1] for x in xs], dtype = torch.long)

	max_t = lengths.max().item()
	padded = []

	for x in xs:
		pad_amt = max_t - x.shape[-1]

		if pad_amt > 0:
			# pad time dim
			x = F.pad(x, (0, pad_amt))

		padded.append(x)

	# (B, 1, F, T)
	x_batch = torch.stack(padded, dim = 0)

	# (B, 1)
	y_batch = torch.stack(ys, dim = 0)

	return x_batch, y_batch, lengths, srs

#-=-=-=-#
# Training / Evaluation

def train_epoch(
	model: nn.Module,
	loader: torch.utils.data.DataLoader,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	custom_name: str
) -> float:
	"""
	Trains a model for one epoch using Huber (smooth L1) loss.

	Parameters
	----------
		model (torch.nn.Module):
			The PyTorch model to train.

		loader (torch.utils.data.DataLoader):
			DataLoader providing input and target batches.

		optimizer (torch.optim.Optimizer):
			Optimizer used to update model parameters.

		device (torch.device):
			Device on which to perform training (e.g., "cpu" or "cuda").

		custom_name (str):
			Name or label shown in the training progress bar.

	Returns
	-------
		float:
			Average loss across the entire dataset for this epoch.

	Example
	-------
		>>> avg_loss = train_epoch(model, train_loader, optimizer, torch.device("cuda"), "Training")
		>>> print(f"Epoch loss: {avg_loss:.4f}")
	"""
	model.train()

	total = 0.0
	criterion = nn.SmoothL1Loss()

	for x, y, lengths, _ in tqdm(loader, desc = custom_name, leave = False):
		x = x.to(device)
		y = y.to(device)
		lengths = lengths.to(device)

		pred = model(x, lengths)
		loss = criterion(pred, y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		total += loss.item() * x.size(0)

	return total / len(loader.dataset)

def eval_epoch(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	custom_name: str
) -> float:
	"""
	Evaluate the model for one epoch using Huber (smooth L1) loss.

	Parameters
	----------
		model (torch.nn.Module):
			The model to evaluate.

		loader (DataLoader):
			DataLoader providing evaluation batches.

		device (torch.device):
			Device to perform computation on.

		custom_name (str):
			Description for progress bar.

	Returns
	-------
		float:
			Average loss over the dataset.
	"""
	model.eval()

	total = 0.0
	criterion = nn.SmoothL1Loss()

	with torch.no_grad():
		for x, y, lengths, _ in tqdm(loader, desc = custom_name, leave = False):
			x = x.to(device)
			y = y.to(device)
			lengths = lengths.to(device)

			pred = model(x, lengths)
			loss = criterion(pred, y)

			total += loss.item() * x.size(0)

	return total / len(loader.dataset)

def fit(
	model: nn.Module,
	train_loader: DataLoader,
	val_loader: DataLoader,
	device: torch.device,

	epochs: int = 100,
	lr: float = 1e-3,
	weight_decay: float = 1e-4,

	checkpoint_path: str = "best_model.pt",

	scheduler_patience: int = 5,
	scheduler_factor: float = 0.5,
	min_lr: float = 1e-6,

	early_stopping_patience: int = 15,
) -> dict:
	"""
	Full training loop with LR scheduling and early stopping.

	Uses ReduceLROnPlateau on validation loss to decay the learning rate
	when progress stalls, and stops training early (with a patience
	window) once validation loss stops improving, checkpointing only the
	best-performing weights rather than the last epoch's.

	Parameters
	----------
		model (nn.Module):
			Model to train (moved to `device` internally if not already).

		train_loader / val_loader (DataLoader):
			Training and validation data loaders (expects pad_collate output).

		device (torch.device):
			Device to train on.

		epochs (int):
			Maximum number of epochs to run.

		lr (float):
			Initial learning rate for Adam.

		weight_decay (float):
			L2 regularization strength.

		checkpoint_path (str):
			Where to save the best model's state_dict.

		scheduler_patience (int):
			Epochs with no val improvement before LR is reduced.

		scheduler_factor (float):
			Multiplicative LR reduction factor when triggered.

		min_lr (float):
			Floor for the LR scheduler.

		early_stopping_patience (int):
			Epochs with no val improvement before training stops.

	Returns
	-------
		dict:
			History with "train_loss" and "val_loss" lists, plus the
			epoch index of the best checkpoint.
	"""
	model = model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode = "min",
		factor = scheduler_factor,
		patience = scheduler_patience,
		min_lr = min_lr,
	)

	best_val_loss = float("inf")
	best_epoch = -1
	epochs_without_improvement = 0

	history = {"train_loss": [], "val_loss": []}

	for epoch in range(1, epochs + 1):
		train_loss = train_epoch(model, train_loader, optimizer, device, f"Epoch {epoch}/{epochs} [train]")
		val_loss = eval_epoch(model, val_loader, device, f"Epoch {epoch}/{epochs} [val]")

		scheduler.step(val_loss)

		history["train_loss"].append(train_loss)
		history["val_loss"].append(val_loss)

		current_lr = optimizer.param_groups[0]["lr"]
		logger.info(
			"Epoch %d/%d - train_loss: %.5f - val_loss: %.5f - lr: %.2e",
			epoch, epochs, train_loss, val_loss, current_lr
		)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_epoch = epoch
			epochs_without_improvement = 0
			torch.save(model.state_dict(), checkpoint_path)
			logger.info("  -> new best val_loss %.5f, checkpoint saved to %s", val_loss, checkpoint_path)
		else:
			epochs_without_improvement += 1

		if epochs_without_improvement >= early_stopping_patience:
			logger.info(
				"Early stopping at epoch %d (no val improvement for %d epochs). Best epoch: %d (val_loss=%.5f).",
				epoch, early_stopping_patience, best_epoch, best_val_loss
			)
			break

	return {**history, "best_epoch": best_epoch, "best_val_loss": best_val_loss}

#-=-=-=-#
# Inference helper

def load_model(
	checkpoint_path: str,
	architecture: nn.Module,

	device: Union[str, torch.device] = "cpu"
) -> nn.Module:
	"""
	Load a trained model from a checkpoint file.

	Parameters
	----------
		checkpoint_path (str):
			Path to the model checkpoint.

		device (Union[str, torch.device]):
			Device to load the model on.

	Returns
	-------
		torch.nn.Module:
			The loaded model in evaluation mode.
	"""
	model = architecture().to(device)
	model.load_state_dict(
		torch.load(checkpoint_path, map_location = device, weights_only = True)
	)

	model.eval()

	return model

def predict(
	audio_path: Union[str, Path, tuple],
	model: Union[nn.Module, str, Path],
	network: nn.Module,
	device: Union[str, torch.device]
) -> float:
	"""
	Predict the cutoff frequency of an audio file or waveform using the model.

	Parameters
	-------
		audio_path (Union[str, Path, tuple]):
			Path to an audio file or a tuple (y, sr).

		model (Union[nn.Module, str, Path]):
			Model instance or path to a checkpoint file.

		network (nn.Module):
			Architecture.

		device (Union[str, torch.device]):
			Device for model inference.

	Returns
	-------
		float:
			Estimated cutoff frequency in Hz.
	"""
	if isinstance(audio_path, str):
		y, sr = librosa.load(audio_path, sr = None, mono = True)
	else:
		y, sr = audio_path # case of tuple

	if isinstance(model, (str, Path)):
		model = load_model(model, network, device)

	model.eval()

	mel = librosa.feature.melspectrogram(y = y, sr = sr, n_mels = 128, n_fft = 2048, hop_length = 512)
	log_mel = librosa.power_to_db(mel, ref = np.max)
	log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

	x = torch.tensor(log_mel, dtype = torch.float32).unsqueeze(0).unsqueeze(0).to(device)

	with torch.no_grad():
		out = model(x).item()

	nyquist = sr / 2.0
	est_cutoff = _target_to_cutoff(out, nyquist)

	return est_cutoff