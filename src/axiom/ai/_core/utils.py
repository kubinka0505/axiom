import os
import torch
import socket
import subprocess
import webbrowser
from typing import Optional
from torch.utils.tensorboard import SummaryWriter

#-=-=-=-#

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val):
	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"scheduler_state_dict": scheduler.state_dict(),
			"epoch": epoch,
			"best_val": best_val,
		}, path
	)

def load_checkpoint(path, model, optimizer = None, scheduler = None, device = "cpu"):
	"""
	Loads either a full training checkpoint (dict with model/optimizer/
	scheduler/epoch/best_val) or a legacy bare state_dict (just weights).

	Returns
	-------
		tuple:
			(start_epoch, best_val)
			(1, inf) for legacy/inference-only loads.
	"""
	loaded = torch.load(path, map_location = device, weights_only = True)

	if isinstance(loaded, dict) and "model_state_dict" in loaded:
		model.load_state_dict(loaded["model_state_dict"])

		if optimizer is not None and "optimizer_state_dict" in loaded:
			optimizer.load_state_dict(loaded["optimizer_state_dict"])

		if scheduler is not None and "scheduler_state_dict" in loaded:
			scheduler.load_state_dict(loaded["scheduler_state_dict"])

		start_epoch = loaded.get("epoch", 0) + 1
		best_val = loaded.get("best_val", float("inf"))

		return start_epoch, best_val

	# legacy: bare state_dict, weights only - no optimizer/epoch/best_val to restore
	model.load_state_dict(loaded)

	return 1, float("inf")

#-=-=-=-#

def find_port() -> int:
	"""
	Find an available ephemeral TCP port on localhost.

	Returns
	-------
		int:
			An unused port number.
	"""
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.bind(("", 0))
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

		return s.getsockname()[1]

def run_tensorboard(
	args,
	port: Optional[int] = None,
	window_title: Optional[str] = None,
	open_web: bool = True,
	logger = None
):
	"""
	Launch TensorBoard pointing to the specified logs directory.

	Parameters
	----------
		args:
			Parsed arguments Namespace containing logs_directory and output_directory.

		port (Optional[int]):
			Port to run TensorBoard on. Finds a free port if None.

		window_title (Optional[str]):
			Window title for TensorBoard.

		open_web (bool):
			Whether to open the TensorBoard URL in a web browser.

	Returns
	-------
		SummaryWriter:
			TensorBoard SummaryWriter instance.
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