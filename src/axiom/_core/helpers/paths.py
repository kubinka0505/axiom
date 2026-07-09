import os
from pathlib import Path

def normalize_path(p: Path, relative: bool = True) -> str:
	"""
	Normalizes a filesystem path relative to the current working directory.

	Expands environment variables and user home (`~`) before resolving to a relative path.

	Parameters
	----------
		p (Path):
			Path object or string representing the file path.

	Returns
	-------
		str:
			Normalized relative path string.

	Example
	-------
		>>> normalize_path(Path("~/Documents/file.txt"))
		'Documents/file.txt'
	"""
	p = str(p)
	p = os.path.expandvars(p)
	p = os.path.expanduser(p)
	p = Path(p)

	try:
		return os.path.relpath(p, Path.cwd()) if relative else p
	except ValueError:
		return str(p.resolve())