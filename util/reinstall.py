import os
import shutil
import subprocess
from pathlib import Path

try:
	import tomllib
except ImportError:
	import tomli as tomllib

#-=-=-=-#

def run(command: list[str]) -> None:
	print(">", " ".join(command))
	subprocess.run(command, check = True)

def remove_dir(path: Path) -> None:
	if path.exists() and path.is_dir():
		print(f"Removing: {path}")
		shutil.rmtree(path)

def find_pyproject(project_root: Path) -> Path | None:
	matches = sorted(project_root.rglob("pyproject.toml"))
	return matches[0] if matches else None

def get_package_name(project_root: Path) -> str:
	pyproject_path = find_pyproject(project_root)

	if pyproject_path and pyproject_path.exists():
		try:
			data = tomllib.loads(pyproject_path.read_text(encoding = "UTF-8"))

			# PEP 621
			name = data.get("project", {}).get("name")
			if name:
				return name

			# poetry
			name = data.get("tool", {}).get("poetry", {}).get("name")
			if name:
				return name

		except Exception as e:
			print(f"Warning: failed to read pyproject.toml: {e}")

	return project_root.name.split()[0].strip()

def main():
	project_root = Path(__file__).resolve().parent.parent
	src_dir = project_root / "src"

	package_name = get_package_name(project_root)

	print(f"Reinstalling module '{package_name}' in editable mode...\n")

	run([os.sys.executable, "-m", "pip", "uninstall", package_name, "-y"])
	run([os.sys.executable, "-m", "pip", "install", "-e", str(project_root)])

	print("\nCleaning build artifacts...")

	for pycache in src_dir.rglob("__pycache__"):
		remove_dir(pycache)

	for egg_info_dir in src_dir.glob("*.egg-info"):
		remove_dir(egg_info_dir)

	remove_dir(src_dir / "build")
	remove_dir(src_dir / "dist")

if __name__ == "__main__":
	main()