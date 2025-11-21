import os
import logging
import argparse
from pathlib import Path

from typing import Optional, Union, List, Tuple

#-=-=-=-#
# Setup

logger = logging.getLogger("axiom-ai")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("[%(levelname)s] %(message)s")

ch = logging.StreamHandler()
ch.setFormatter(formatter)

logger.addHandler(ch)

EXTENSIONS_AUDIO = "WAV", "FLAC"
EXTENSIONS_CKPTS = "PT", "PTH"

EXTENSIONS_AUDIO = ["." + ext for ext in EXTENSIONS_AUDIO]
EXTENSIONS_CKPTS = ["." + ext for ext in EXTENSIONS_CKPTS]

#-=-=-=-#
# Argparse

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
	pass

def parse_args():
	parser = argparse.ArgumentParser(
		description = "Train cutoff frequency regressor or run inference.",
		formatter_class = CustomFormatter,
		add_help = False
	)

	parser_switch = parser.add_argument_group("Switch arguments")

	#-=-=-=-#

	subparsers = parser.add_subparsers(
		dest = "mode",
		required = True,
		help = "Mode to run."
	)

	train_p = subparsers.add_parser(
		"train",
		help = "Train model.",
		formatter_class = CustomFormatter
	)

	infer_p = subparsers.add_parser(
		"infer",
		help = "Run inference.",
		formatter_class = CustomFormatter
	)

	#-=-=-=-#
	# Switch

	parser_switch.add_argument(
		"-h", "--help",
		action = "help",
		help = "Shows this message."
	)


	#-=-=-=-#
	# --- Train subcommand ---

	# Groups
	train_req = train_p.add_argument_group("Required arguments")
	train_opt = train_p.add_argument_group("Optional arguments")
	train_switch = train_p.add_argument_group("Switch arguments")

	#-=-=-=-#
	# Required

	train_req.add_argument(
		"-i", "--dataset",
		type = str,
		required = True,
		metavar = '"str"',
		help = "Directory with audio files."
	)

	train_req.add_argument(
		"-e", "--epochs",
		type = int,
		required = True,
		metavar = "int",
		help = "Number of epochs to train."
	)

	#-=-=-=-#
	# Optional

	train_opt.add_argument(
		"-a", "--architecture",
		type = int,
		metavar = "int",
		choices = [1, 2],
		default = 2,
		help = "Model architecture to train."
	)

	train_opt.add_argument(
		"-l", "--labels",
		type = str,
		default = None,
		metavar = '"str"',
		help = "Path to labels mapping file. If omitted, picks first .json in dataset."
	)

	train_opt.add_argument(
		"-o", "--output-directory",
		type = str,
		default = os.path.join("logs", "models"),
		metavar = '"str"',
		help = "Checkpoint output directory."
	)

	train_opt.add_argument(
		"-bs", "--batch-size",
		type = int,
		default = 8,
		metavar = "int",
		help = "Batch size for training/validation."
	)

	train_opt.add_argument(
		"-lr", "--learning-rate",
		type = float,
		default = "2e-4",
		metavar = "float",
		help = "Learning rate for optimizer."
	)

	train_opt.add_argument(
		"-see", "--save-each-epoch",
		type = int,
		metavar = "int",
		default = 10,
		help = "Save checkpoint every this many epochs."
	)

	train_opt.add_argument(
		"-eee", "--evaluate-each-epoch",
		type = int,
		metavar = "int",
		default = 1,
		help = "Evaluate checkpoint every this many epochs."
	)

	train_opt.add_argument(
		"-p", "--patience",
		type = int,
		metavar = "int",
		default = 10,
		help = "Early stopping patience (no improvement)."
	)

	train_opt.add_argument(
		"-logs", "--logs-directory",
		type = str,
		metavar = '"str"',
		default = "logs",
		help = "TensorBoard logs directory."
	)

	#-=-=-=-#
	# Switch

	train_switch.add_argument(
		"-ntb", "--no-tensorboard",
		action = "store_true",
		help = "Disable TensorBoard during training."
	)

	#-=-=-=-#
	# --- Infer subcommand ---

	# Groups
	infer_req = infer_p.add_argument_group("Required arguments")

	#-=-=-=-#
	# Required
	infer_req.add_argument(
		"-i", "--input",
		type = str,
		metavar = '"str"',
		required = True,
		help = "Path to audio file for inference."
	)

	infer_req.add_argument(
		"-m", "--checkpoint",
		type = str,
		metavar = '"str"',
		required = True,
		help = "Checkpoint to load."
	)

	#-=-=-=-#
	# Actions

	for action in infer_req._group_actions:
		action.required = True

	for action in train_req._group_actions:
		action.required = True

	#-=-=-=-#

	args = parser.parse_args()

	# Validation for train
	if args.mode.lower() == "train":
		if not args.dataset:
			parser.error("No dataset directory provided")

		if not os.path.isdir(args.dataset):
			parser.error("Dataset directory is invalid")

		if args.epochs is None or args.epochs <= 0:
			parser.error("Epochs must be > 0")

		if not args.labels:
			labels = list(Path(args.dataset).glob("*.json"))

			if labels:
				args.labels = str(labels[0])

		if not args.labels:
			parser.error("No labels file has been found")

		if args.batch_size <= 0:
			parser.error("Batch size must be > 0")

		try:
			args.learning_rate = float(args.learning_rate)
		except ValueError:
			parser.error("Couldn't parse learning rate value")

		if args.save_each_epoch > args.epochs:
			args.save_each_epoch = args.epochs

		if args.save_each_epoch <= 0:
			parser.error("Save each epoch must be > 0")

		if args.evaluate_each_epoch <= 0:
			parser.error("Eval each epoch must be > 0")

	if args.mode.lower() == "infer":
		if os.path.isdir(args.input):
			args.input = Path(args.input).glob("*.*")
			args.input = [str(file.resolve()) for file in args.input]
			args.input = [file for file in args.input if file.upper().endswith(EXTENSIONS_AUDIO)]
			args.input = sorted(args.input, os.path.getmtime)

			if args.input:
				args.input = args.input[-1]
			else:
				args.input = None

		if not args.input:
			parser.error("Input file was not found")

		#-=-=-=-#

		if os.path.isdir(args.checkpoint):
			args.checkpoint = Path(args.checkpoint).glob("*.*")
			args.checkpoint = [str(file.resolve()) for file in args.checkpoint]
			args.checkpoint = [file for file in args.checkpoint if file.upper().endswith(EXTENSIONS_CKPTS)]
			args.checkpoint = sorted(args.checkpoint, os.path.getmtime)

			if args.checkpoint:
				args.checkpoint = args.checkpoint[-1]
			else:
				args.checkpoint = None

		if not args.checkpoint:
			parser.error("Input file was not found")

	return args

args = parse_args()

import json
import time
import socket
import datetime
import warnings
import subprocess
import webbrowser
from tqdm import tqdm

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split

warnings.filterwarnings("ignore", category = UserWarning, message = "TypedStorage is deprecated")

#-=-=-=-#
# Utils

def find_port() -> int:
	"""
	Find an available ephemeral TCP port on localhost.

	Returns:
		int: An unused port number.
	"""
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.bind(("", 0))
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

		return s.getsockname()[1]

def run_tensorboard(
	args,
	port: Optional[int] = None,
	window_title: Optional[str] = None,
	open_web: bool = True
) -> SummaryWriter:
	"""
	Launch TensorBoard pointing to the specified logs directory.

	Args:
		args: Parsed arguments containing logs_directory and output_directory.
		port (Optional[int]): Port to run TensorBoard on. Finds a free port if None.
		window_title (Optional[str]): Window title for TensorBoard.
		open_web (bool): Whether to open the TensorBoard URL in a web browser.

	Returns:
		SummaryWriter: TensorBoard SummaryWriter instance.
	"""
	if not port:
		port = find_port()

	if not window_title:
		window_title = "TensorBoard"

	writer = SummaryWriter(log_dir = args.logs_directory)

	port = find_port()
	url = f"http://localhost:{port}"

	logger.info(f"Running TensorBoard at: {url} (Press CTRL + C to quit)")
	subprocess.Popen(
		[
			"tensorboard",
			"--logdir", args.logs_directory,
			"--host", "localhost",
			"--port", str(port),
			"--window_title", os.path.basename(args.output_directory)
		],
		stdout = subprocess.DEVNULL,
		stderr = subprocess.DEVNULL
	)

	if open_web:
		webbrowser.open(url + "?darkMode=true&smoothing=0.99#scalars&_smoothingWeight=0.999")

	return writer

#-=-=-=-#
# Architectures

class Architectures:
	class CutoffNetV1(nn.Module):
		"""
		A simple convolutional neural network for binary classification.

		The architecture consists of three convolutional blocks with batch normalization and ReLU activations, followed by max pooling.
		Features are globally averaged and passed through a fully connected layer with a sigmoid activation to produce the final output.

		Designed for 1D input.
		"""
		def __init__(self):
			super().__init__()

			self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, padding = 1)
			self.bn1 = nn.BatchNorm2d(16)
			self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)
			self.bn2 = nn.BatchNorm2d(32)
			self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
			self.bn3 = nn.BatchNorm2d(64)

			self.avgpool = nn.AdaptiveAvgPool2d((1,1))
			self.fc = nn.Linear(64, 1)

		def forward(self, x: torch.Tensor):
			"""
			Defines the forward pass of the network.

			Args:
				x (torch.Tensor): Input tensor of shape (N, 1, H, W), where N is batch size.

			Returns:
				torch.Tensor: Output tensor of shape (N, 1) with values in range [0, 1], 
				representing predicted probabilities for binary classification.

			Example:
				>>> model = CutoffNetV1()
				>>> input_tensor = torch.randn(8, 1, 64, 64)
				>>> output = model(input_tensor)
				>>> output.shape
				torch.Size([8, 1])
			"""
			x = F.relu(self.bn1(self.conv1(x)))
			x = F.max_pool2d(x, 2)
			x = F.relu(self.bn2(self.conv2(x)))
			x = F.max_pool2d(x, 2)
			x = F.relu(self.bn3(self.conv3(x)))

			x = self.avgpool(x)
			x = x.view(x.size(0), -1)
			x = torch.sigmoid(self.fc(x))

			return x

	class CutoffNetV2(nn.Module):
		"""
		A deeper convolutional neural network for estimating normalized cutoff frequency from mel spectrograms.

		Architecture:
		- 4 convolutional blocks (Conv2d → BatchNorm → ReLU → MaxPool or AdaptiveAvgPool)
		- Followed by dropout, two fully connected (FC) layers, and ReLU
		- Final output is a raw float (no sigmoid), representing the normalized cutoff frequency [0.0, 1.0]

		Input:
		- Tensor of shape (B, 1, 128, T), where 128 is the mel frequency bins and T is time frames

		Output:
		- Tensor of shape (B, 1), with predicted normalized cutoff values (not thresholded)

		Use with MSELoss for regression.
		"""
		def __init__(self):
			super().__init__()

			self.net = nn.Sequential(
				nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
				nn.BatchNorm2d(32),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size = 2),

				nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
				nn.BatchNorm2d(64),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size = 2),

				nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
				nn.BatchNorm2d(128),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size = 2),

				nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
				nn.BatchNorm2d(128),
				nn.ReLU(),
				nn.AdaptiveAvgPool2d((1, 1)),

				nn.Flatten(),
				nn.Dropout(0.2),
				nn.Linear(128, 64),
				nn.ReLU(),
				nn.Dropout(0.2),
				nn.Linear(64, 1)
			)

		def forward(self, x: torch.Tensor) -> torch.Tensor:
			"""
			Forward pass through the network.

			Args:
				x (torch.Tensor): Input tensor of shape (B, 1, 128, T), where:
					- B is batch size
					- 1 is the input channel (mel spectrogram)
					- 128 is the mel bins (frequency dimension)
					- T is variable-length time dimension (can be padded)

			Returns:
				torch.Tensor: Output tensor of shape (B, 1) representing normalized cutoff predictions.
			"""
			return self.net(x)

#-=-=-=-#
# Dataset

class CutoffDataset(Dataset):
	def __init__(self, json_map_path: str, audio_dir: str, sample_length: Optional[float] = None):
		"""
		Dataset for audio files mapped to cutoff frequencies.

		Args:
			json_map_path (str): Path to JSON file mapping filenames to cutoff frequencies.
			audio_dir (str): Directory containing audio files.
			sample_length (Optional[float]): Fixed length of audio samples in seconds; if None, use full length.
		"""
		with open(json_map_path, "r") as f:
			self.mapping = json.load(f)

		self.audio_dir = audio_dir
		self.keys = list(self.mapping.keys())
		self.sample_length = sample_length # seconds

	def __len__(self):
		"""
		Return number of audio samples.
		"""
		return len(self.keys)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
		"""
		Get the mel spectrogram, target cutoff frequency, and sample rate for an audio sample.

		Args:
			idx (int): Index of the sample.

		Returns:
			Tuple containing:
				- x (torch.Tensor): Mel spectrogram tensor of shape (1, freq, time).
				- y_target (torch.Tensor): Target cutoff frequency normalized (scalar tensor).
				- sr (int): Sample rate of the audio.
		"""
		fname = self.keys[idx]
		cutoff_hz = self.mapping[fname]
		path = os.path.join(self.audio_dir, fname)

		y, sr = librosa.load(path, sr = None)

		if self.sample_length is not None:
			target_len = int(self.sample_length * sr)

			if len(y) < target_len:
				y = np.pad(y, (0, target_len - len(y)))
			else:
				y = y[:target_len]

		mel = librosa.feature.melspectrogram(y = y, sr = sr, n_mels = 128, n_fft = 2048, hop_length = 512)
		log_mel = librosa.power_to_db(mel, ref = np.max)

		# per-sample standardization
		log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1E-6)
		x = torch.tensor(log_mel, dtype = torch.float32).unsqueeze(0) # (1, freq, time)

		nyquist = sr / 2.0
		target = cutoff_hz / nyquist
		target = np.clip(target, 0.0, 1.0)

		y_target = torch.tensor([target], dtype = torch.float32)

		return x, y_target, sr

def pad_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
	"""
	Pads variable-length tensors in a batch along the time dimension for DataLoader compatibility.

	This function is used as a custom `collate_fn` in a PyTorch DataLoader.
	It takes a list of (input, target, sample_rate) tuples and zero-pads the input tensors along the time dimension so they can be stacked into a single batch tensor.

	Assumes each input `x` has shape (1, freq, time), and all inputs have the same frequency dimension.

	Args:
		batch (list of tuples): A list of samples, where each sample is a tuple (x, y, sr):
			- x (torch.Tensor): Input tensor of shape (1, freq, time).
			- y (torch.Tensor): Target tensor (any shape).
			- sr (int): Sample rate (passed through unchanged).

	Returns:
		tuple:
			- x_batch (torch.Tensor): Batched and padded input tensor of shape (B, 1, freq, max_time).
			- y_batch (torch.Tensor): Batched target tensor.
			- srs (tuple): Tuple of original sample rates for each sample in the batch.

	Example:
		>>> loader = DataLoader(dataset, batch_size = 8, collate_fn = pad_collate)
	"""
	# batch: list of (x, y, sr)
	xs, ys, srs = zip(*batch)

	# determine max time dimension in this batch
	max_t = max(x.shape[-1] for x in xs) # x is (1, freq, time)
	padded = []

	for x in xs:
		pad_amt = max_t - x.shape[-1]

		if pad_amt > 0:
			# pad on right (time axis) with zeros: F.pad uses (left, right) per dim from last
			x = F.pad(x, (0, pad_amt))

		padded.append(x)

	x_batch = torch.stack(padded, dim = 0) # (B,1,freq,max_t)
	y_batch = torch.stack(ys, dim = 0)

	return x_batch, y_batch, srs

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
	Trains a model for one epoch using mean squared error loss.

	This function performs one full training pass over the dataset using the provided DataLoader, optimizer, and model.
	Loss is accumulated and returned as the average loss over the dataset.

	Args:
		model (torch.nn.Module): The PyTorch model to train.
		loader (torch.utils.data.DataLoader): DataLoader providing input and target batches.
		optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
		device (torch.device): Device on which to perform training (e.g., "cpu" or "cuda").
		custom_name (str): Name or label shown in the training progress bar.

	Returns:
		float: Average loss across the entire dataset for this epoch.

	Example:
		>>> avg_loss = train_epoch(model, train_loader, optimizer, torch.device("cuda"), "Training")
		>>> print(f"Epoch loss: {avg_loss:.4f}")
	"""
	model.train()

	total = 0.0
	criterion = nn.MSELoss()

	for x, y, _ in tqdm(loader, desc = custom_name, leave = False):
		x = x.to(device)
		y = y.to(device)

		pred = model(x)
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
	Evaluate the model for one epoch using MSE loss.

	Args:
		model (torch.nn.Module): The model to evaluate.
		loader (DataLoader): DataLoader providing evaluation batches.
		device (torch.device): Device to perform computation on.
		custom_name (str): Description for progress bar.

	Returns:
		float: Average loss over the dataset.
	"""
	model.eval()

	total = 0.0
	criterion = nn.MSELoss()

	with torch.no_grad():
		for x, y, _ in tqdm(loader, desc = custom_name, leave = False):
			x = x.to(device)
			y = y.to(device)

			pred = model(x)
			loss = criterion(pred, y)

			total += loss.item() * x.size(0)

	return total / len(loader.dataset)

#-=-=-=-#
# Inference helper

def load_model(checkpoint_path: str, architecture: nn.Module = Architectures.CutoffNetV2, device: Union[str, torch.device] = "cpu") -> nn.Module:
	"""
	Load a trained model from a checkpoint file.

	Args:
		checkpoint_path (str): Path to the model checkpoint.
		device (Union[str, torch.device]): Device to load the model on.

	Returns:
		torch.nn.Module: The loaded model in evaluation mode.
	"""
	model = architecture().to(device)
	model.load_state_dict(
		torch.load(checkpoint_path, map_location = device)
	)
	model.eval()

	return model

def predict_cutoff(
	audio_path: Union[str, Path, tuple],
	model: Union[nn.Module, str, Path],
	network: nn.Module,
	device: Union[str, torch.device]
) -> float:
	"""
	Predict the cutoff frequency of an audio file or waveform using the model.

	Args:
		audio_path (Union[str, Path, tuple]): Path to an audio file or a tuple (y, sr).
		model (Union[nn.Module, str, Path]): Model instance or path to a checkpoint file.
		network (nn.Module): Architecture.
		device (Union[str, torch.device]): Device for model inference.

	Returns:
		float: Estimated cutoff frequency in Hz.
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
	log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1E-6)

	x = torch.tensor(log_mel, dtype = torch.float32).unsqueeze(0).unsqueeze(0).to(device)

	with torch.no_grad():
		out = model(x).item()

	est_cutoff = out * (sr / 2)

	return est_cutoff

#-=-=-=-#

def main():
	device = "cuda" if torch.cuda.is_available() else "cpu"
	if device == "cpu":
		logger.warning("GPU not found, defaulting to CPU. Processing may be slow!")

	if args.mode.lower() == "infer":
		if not args.checkpoint:
			raise RuntimeError("Need --checkpoint to do inference")

		loaded = torch.load(args.checkpoint, map_location = device, weights_only = True)

		# Detect architecture based on state_dict keys
		if any("conv1.weight" in k for k in loaded.keys()):
			network = Architectures.CutoffNetV1 # looks like conv1.weight means v1
		else:
			network = Architectures.CutoffNetV2 # else assume v2

		model = network().to(device)
		model.load_state_dict(loaded) # load actual state dict

		est = predict_cutoff(args.input, model, network, device)
		print(est)

		return est
	else:
		if args.architecture == 1:
			network = Architectures.CutoffNetV1
		elif args.architecture == 2:
			network = Architectures.CutoffNetV2

	# train path
	os.makedirs(args.output_directory, exist_ok = True)

	name_loss_train = "loss/train"
	name_loss_val = "loss/val"
	name_loss_best_val = "loss/best_val"

	#-=-=-=-#
	# training path

	logger.info("Running CutoffDataset...")
	full = CutoffDataset(args.labels, args.dataset)

	val_size = max(1, int(0.2 * len(full)))
	train_size = len(full) - val_size
	train_ds, val_ds = random_split(full, [train_size, val_size])

	logger.info("Loading train_loader...")
	train_loader = DataLoader(
		train_ds,
		batch_size = args.batch_size,
		shuffle = True,
		collate_fn = pad_collate
	)

	logger.info("Loading val_loader...")
	val_loader = DataLoader(
		val_ds,
		batch_size = args.batch_size,
		shuffle = False,
		collate_fn = pad_collate
	)

	logger.info(f"Moving {network.__name__} to {device.upper()}...")
	model = network().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

	best_val = float("inf")
	epochs_no_improve = 0

	start_time = time.time()
	train_prefix = "[TRAIN]"

	try:
		for epoch in range(1, args.epochs + 1):
			if not args.no_tensorboard and epoch == 1:
				logger.info("Loading TensorBoard...")
				port = find_port()
				writer = run_tensorboard(args = args, port = port)

			step = epoch * len(train_loader)

			train_loss = train_epoch(model, train_loader, optimizer, device, f"{train_prefix} Epoch {epoch} Step {step}")

			if epoch == 1 or not epoch % args.evaluate_each_epoch:
				val_loss = eval_epoch(model, val_loader, device, f"{train_prefix} Evaluating")

			architecture = network.__name__[-1].lower()

			ckpt_name = f"axiom_cutoff_{epoch}e_{step}s_v{architecture}.pt"
			ckpt_path = os.path.join(args.output_directory, ckpt_name)

			try:
				writer.add_scalar(name_loss_train, train_loss, epoch)
				writer.add_scalar(name_loss_val, val_loss, epoch)
				writer.add_scalar(name_loss_best_val, best_val, epoch)
			except (NameError, UnboundLocalError):
				pass

			epoch_start_time = time.time()

			if val_loss < best_val:
				best_val = val_loss
				epochs_no_improve = 0

				logger.info(" ".join((
					f"epoch={epoch}",
					f"step={step}",
					" ",
					f"train MSE={train_loss:.6f}",
					f"val MSE={val_loss:.6f}",
					" ",
					"runtime={0}".format(
						str(datetime.timedelta(seconds = time.time() - epoch_start_time))[2:-3]
					)
				)))

				if not epoch % args.save_each_epoch:
					torch.save(
						model.state_dict(),
						ckpt_path
					)
			else:
				epochs_no_improve += 1

			if val_loss < best_val:
				best_val = val_loss

				torch.save(
					model.state_dict(),
					os.path.join(args.output_directory, f"best_{epoch}e_{step}s_v{architecture}.pt")
				)

			if epochs_no_improve >= args.patience:
				logger.warning("No improvement for {args.patience} epochs. Early stopping.")
				break
	except KeyboardInterrupt:
		logger.warning("Operation interrupted by user.")
		epoch -= 1
	finally:
		try:
			if not args.no_tensorboard:
				writer.flush()
				writer.close()
		except UnboundLocalError:
			pass

	end_time = str(datetime.timedelta(seconds = time.time() - start_time))[2:-3]
	logger.info(f"Training {epoch} epochs done in {end_time}.\a")
	logger.info(f"Checkpoint saved to {ckpt_path}.")

if __name__ == "__main__":
	main()