from ..core import *
from ..utils import *
from ..vars import *

#-=-=-=-#

import os
import time
import torch
import datetime
from contextlib import suppress

def main(args):
	device = "cuda" if torch.cuda.is_available() else "cpu"

	if device == "cpu":
		logger.warning("GPU not found, defaulting to CPU. Processing WILL be slow!")

	network = Architectures.CutoffNet

	#-=-=-=-#
	# inference path
	if args.mode.lower() == "infer":
		if not args.file_checkpoint:
			raise RuntimeError("Need checkpoint to do inference")

		model = network().to(device)

		loaded = torch.load(
			args.file_checkpoint,
			map_location = device,
			weights_only = True
		)

		model.load_state_dict(loaded)
		model.eval()

		est = inference(args.file_input, model, network, device)
		print(est)

		return est

	#-=-=-=-#
	# training path
	os.makedirs(args.output_directory, exist_ok = True)

	name_loss_train = "loss/train"
	name_loss_val = "loss/val"
	name_loss_best_val = "loss/best_val"

	logger.info("Running dataset cutoff...")
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

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode = "min", factor = 0.5, patience = max(2, args.patience // 3)
	)

	architecture_tag = network.__name__

	start_epoch = 1
	best_val = float("inf")

	if args.file_input:
		logger.info(f"Resuming training from {args.file_input}...")

		start_epoch, best_val = load_checkpoint(
			args.file_input,
			model,
			optimizer,
			scheduler,
			device
		)
		logger.info(f"Resuming at epoch {start_epoch}, best_val so far = {best_val:.6f}")

	best_val = float("inf")
	epochs_no_improve = 0
	val_loss = None

	start_time = time.time()
	train_prefix = "[TRAIN]"

	writer = None
	ckpt_path = None
	epoch = 0

	try:
		for epoch in range(1, args.epochs + 1):
			epoch_start_time = time.time()

			# TensorBoard init (safe)
			if not args.no_tensorboard and epoch == 1:
				logger.info("Loading TensorBoard...")
				port = find_port()
				writer = run_tensorboard(args = args, port = port, logger = logger)

			step = epoch * len(train_loader)

			#-=-=-=-#
			# TRAIN
			train_loss = train_epoch(
				model,
				train_loader,
				optimizer,
				device,
				f"{train_prefix} Epoch {epoch} Step {step}"
			)

			#-=-=-=-#
			# VALIDATION (always evaluate for early stopping correctness)
			should_eval = (epoch == 1) or (epoch % args.evaluate_each_epoch == 0)

			if should_eval:
				val_loss = eval_epoch(
					model,
					val_loader,
					device,
					f"{train_prefix} Evaluating"
				)

				scheduler.step(val_loss)

			ckpt_path = os.path.join(
				args.output_directory,
				f"cutoff_{epoch}e_{step}s_{architecture_tag}.pt"
			)

			#-=-=-=-#
			# TensorBoard logging
			if writer is not None:
				writer.add_scalar(name_loss_train, train_loss, epoch)
				writer.add_scalar(name_loss_val, val_loss, epoch)
				writer.add_scalar(name_loss_best_val, best_val, epoch)

			epoch_runtime = str(datetime.timedelta(seconds = time.time() - epoch_start_time))[2:-3]

			#-=-=-=-#
			# CHECKPOINT (best) + EARLY STOPPING LOGIC
			if val_loss < best_val:
				best_val = val_loss
				epochs_no_improve = 0

				logger.info(
					" ".join((
						f"epoch={epoch}",
						f"step={step}",
						f"train loss={train_loss:.6f}",
						f"val loss={val_loss:.6f}",
						f"runtime={epoch_runtime}"
					))
				)

				torch.save(
					model.state_dict(),
					os.path.join(
						args.output_directory,
						f"best_{epoch}e_{step}s_{architecture_tag}.pt"
					)
				)
			else:
				epochs_no_improve += 1

				logger.info(
					f"epoch={epoch} step={step} "
					f"train loss={train_loss:.6f} val loss={val_loss:.6f} "
					f"(no improvement, {epochs_no_improve}/{args.patience}) "
					f"runtime={epoch_runtime}"
				)

			if args.save_each_epoch and not epoch % args.save_each_epoch:
				torch.save(model.state_dict(), ckpt_path)

			#-=-=-=-#
			# EARLY STOPPING
			if epochs_no_improve >= args.patience:
				logger.warning(f"No improvement for {args.patience} epochs. Early stopping.")
				break
	except KeyboardInterrupt:
		logger.warning("Operation interrupted by user.")
		epoch = max(0, epoch - 1)
	finally:
		if writer is not None:
			with suppress(OSError):
				writer.flush()
				writer.close()

	#-=-=-=-#
	# END LOGGING

	end_time = str(
		datetime.timedelta(seconds = time.time() - start_time)
	)[2:-3]

	logger.info(f"Training {epoch} epochs done in {end_time}.\a")

	if ckpt_path:
		logger.info(f"Last checkpoint path: {ckpt_path}")