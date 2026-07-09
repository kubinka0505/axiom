from ..setup import EXTENSIONS_DISPLAY

import os
from pathlib import Path

def file_size(obj) -> str:
	"""
	Outputs the size of the file at the given object in a human-readable format.
	
	Parameters
	----------
		obj:
			The path to the file or bytes.
	
	Returns
	-------
		str:
			File size as a string with appropriate unit (e.g., "10.5 MB").
	
	Raises
	------
		FileNotFoundError:
			If the file could not be found.

		ValueError:
			If the path is not a file.
	"""
	if isinstance(obj, str):
		if not os.path.exists(obj):
			raise FileNotFoundError(f'The file at path "{obj}" does not exist.')

		if not os.path.isfile(obj):
			raise ValueError(f'The path "{obj}" is not a file.')

		size_bytes = os.path.getsize(obj)
	else:
		size_bytes = obj

	units = ["B", "KB", "MB", "GB", "TB", "PB"]
	index = 0

	while size_bytes >= 1024 and index < len(units) - 1:
		size_bytes /= 1024.0
		index += 1

	if size_bytes <= 1024:
		size_bytes = f"{size_bytes:.2f}"
	else:
		size_bytes = str(int(size_bytes))

	return " ".join((size_bytes, units[index]))

def package_search(query: str, variable: str = "PATH") -> str:
	"""
	Search for a package.

	Parameters
	----------
		query (str)

		variable (str):
			Environment variable to look for package.

	Windows
	-------
		On windows, searches variable's entries for PE executables (MZ header).

	Linux
	-----
		Checks whether an APT package is installed.

	Returns
	-------
		str:
			Absolute path to the executable/package name.
			Empty string if nothing was found.
	"""
	query = query.lower().strip()

	if not query:
		return ""

	# Windows
	if os.name == "nt":
		for entry in os.environ.get(variable, "").split(os.pathsep):
			entry = entry.strip()

			if not entry:
				continue

			try:
				path = Path(entry)

				if not path.exists():
					continue

				if query not in str(path).lower():
					continue

				candidates = (
					[p for p in path.iterdir() if p.is_file()]
					if path.is_dir()
					else [path]
				)

				for candidate in candidates:
					try:
						with candidate.open("rb") as f:
							if f.read(2) == b"MZ":
								return str(candidate.resolve())
					except (PermissionError, OSError):
						continue

			except (PermissionError, OSError):
				continue

		return ""

	# Linux / Debian-based
	try:
		import apt
		cache = apt.Cache()

		if query in cache and cache[query].is_installed:
			return query

	except (ImportError, KeyError):
		pass

	return ""

def choose_files_dialog(initialdir: str = os.getcwd()) -> list:
	"""
	Open a files selection dialog for choosing audio files.

	Parameters
	----------
		initialdir (str):
			Initial directory to open.

	Returns
	-------
		str:
			Normalized path to the selected files.

	Raises
	------
		Exception:
			If no files are selected or if dialog fails.
	"""
	from tkinter import Tk, filedialog

	root = Tk()
	root.withdraw()

	file_paths = filedialog.askopenfilenames(
		title = "Select audio files",
		initialdir = initialdir,
		filetypes = [
			("Audio files", " ".join(list(EXTENSIONS_DISPLAY.values()))),
			*list(EXTENSIONS_DISPLAY.items()),
			("All files", "*.*"),
		],
		defaultextension = "*.wav"
	)

	if not file_paths:
		file_paths = filedialog.askdirectory(
			title = "Select directory with audio files",
			initialdir = initialdir
		)

	file_paths = [os.path.normpath(file) for file in file_paths]

	return file_paths