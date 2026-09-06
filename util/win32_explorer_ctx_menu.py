import os
os.environ["PYTHON_REGISTRY_UAC"] = ""

try:
	import tomllib
except ImportError:
	import tomli as tomllib

import regmgr

#-=-=-=-#
# Variables

_fmts = "WAV", "FLAC", "MP3", "OGG"
_fmts_disp = [f".{fmt.lower()}" for fmt in _fmts]

try:
	name_prog = tomllib.load(
		open(os.path.join(os.pardir, "pyproject.toml"), "rb")
	)["project"]["name"]
except FileNotFoundError:
	name_prog = os.path.basename(os.path.dirname(__file__))

loc_py = os.sys.executable

name_ctxMenus = "ContextMenus"
name_commands = "Commands"

name_analyze = "Analyze"
name_process = "Process"
name_globalPipeline = "GlobalPipeline"

#-=-=-=-#
# Uninstall

shell_global = regmgr.RegEntry(os.path.join("HKCU", "Software", "Classes", "*", "Shell", name_prog))
shell_hooks = regmgr.RegEntry(os.path.join("HKCU", "Software", "Classes", "*", name_ctxMenus, name_prog))

if not "-f" in os.sys.argv and (shell_global.exists() or shell_hooks.exists()):
	print("Registry keys are already installed.\a\n")

	while True:
		try:
			ip = input("Uninstall? [y/n]: ")
		except (EOFError, KeyboardInterrupt):
			pass

		if ip.lower() == "y":
			shell_global.remove_subkeys()
			shell_hooks.remove_subkeys()

			print()
			raise SystemExit("Uninstalled successfully\a")
		elif ip.lower() == "n":
			print()
			break

#-=-=-=-#
# Global
shell_global = regmgr.RegEntry(os.path.join("HKCU", "Software", "Classes", "*", "Shell", name_prog))

shell_global.make()

shell_global[""] = f"&{name_prog}"
shell_global["AppliesTo"] = " OR ".join(f"System.FileExtension:={ext}" for ext in _fmts_disp).strip()
shell_global["ExtendedSubCommandsKey"] = os.path.join("*", name_ctxMenus, name_prog, name_commands)
shell_global["Icon"] = os.path.abspath(os.path.join(os.pardir, "docs", "img", "Icon.ico"))
shell_global["MultiSelectModel"] = "Document"

#-=-=-=-#
# Hooks
shell_hooks = regmgr.RegEntry(os.path.join("HKCU", "Software", "Classes", "*", name_ctxMenus, name_prog))
shell_hooks.make()

shell_hooks.relcd(name_commands)
shell_hooks.make(name_analyze)
shell_hooks.make("Process")
shell_hooks.make("Shell")





#-=-=-=-#
# Analyze commands

# command 1
shell_hooks.relcd(os.path.join(name_analyze, name_analyze + name_globalPipeline))
shell_hooks.relcd(os.path.join("Shell", "01_ALL_AIO"))
shell_hooks.make()

shell_hooks[""] = "All"
shell_hooks["Icon"] = ",".join(("Shell32.dll", "-16763"))

shell_hooks.relcd("command")
shell_hooks.make()

shell_hooks[""] = f'"{loc_py}" -m {name_prog} -i "%1" -v 2'
shell_hooks.relcd(os.path.join(os.pardir, os.pardir)) # exit current command

#---
# exit current command subkey
shell_hooks.relcd(os.path.join(os.pardir, os.pardir))



#---
# exit this commands section
shell_hooks.relcd(os.pardir)


#-----#


#-=-=-=-#
# Process commands

# command 1
shell_hooks.relcd(os.path.join(name_process, name_process + name_globalPipeline))
shell_hooks.relcd(os.path.join("Shell", "01_SG100"))
shell_hooks.make()

shell_hooks[""] = "Spectral Gate (Fullband, -100 dB) → Channel Detection → FLAC"
shell_hooks["Icon"] = ",".join(("NetCenter.dll", "-19"))

shell_hooks.relcd("command")
shell_hooks.make()

shell_hooks[""] = f'"{loc_py}" -m {name_prog} -i "%1" -f -o "%1.flac" -gc 100 -ol "logs" -nsr -nbd -nc -nbr -np -v -1'
shell_hooks.relcd(os.path.join(os.pardir, os.pardir)) # exit current command

#---
# exit current command subkey
shell_hooks.relcd(os.path.join(os.pardir, os.pardir))

#---
# exit this commands section
shell_hooks.relcd(os.pardir)

#-----#

# command 2
shell_hooks.relcd(os.path.join(name_process, name_process + name_globalPipeline))
shell_hooks.relcd(os.path.join("Shell", "02_SG90"))
shell_hooks.make()

shell_hooks[""] = "Spectral Gate (Fullband, -90 dB) → Channel Detection"
shell_hooks["Icon"] = ",".join(("NetCenter.dll", "-18"))

shell_hooks.relcd("command")
shell_hooks.make()

shell_hooks[""] = f'"{loc_py}" -m {name_prog} -i "%1" -f -o "%1" -gc 90 -nsr -nbd -nc -nbr -np -v -1'
shell_hooks.relcd(os.path.join(os.pardir, os.pardir)) # exit current command

#---
# exit current command subkey
shell_hooks.relcd(os.path.join(os.pardir, os.pardir))



#---
# exit this commands section
shell_hooks.relcd(os.pardir)

#-----#

# command 3
shell_hooks.relcd(os.path.join(name_process, name_process + name_globalPipeline))
shell_hooks.relcd(os.path.join("Shell", "03_SG80"))
shell_hooks.make()

shell_hooks[""] = "Spectral Gate (Fullband, -80 dB) → Channel Detection"
shell_hooks["Icon"] = ",".join(("NetCenter.dll", "-17"))

shell_hooks.relcd("command")
shell_hooks.make()

shell_hooks[""] = f'"{loc_py}" -m {name_prog} -i "%1" -f -o "%1" -gc 80 -nsr -nbd -nbr -np -v -1'
shell_hooks.relcd(os.path.join(os.pardir, os.pardir)) # exit current command

#---
# exit current command subkey
shell_hooks.relcd(os.path.join(os.pardir, os.pardir))



#---
# exit this commands section
shell_hooks.relcd(os.pardir)

#-----#

# command 4
shell_hooks.relcd(os.path.join(name_process, name_process + name_globalPipeline))
shell_hooks.relcd(os.path.join("Shell", "04_SG_NORM0"))
shell_hooks.make()

shell_hooks[""] = "Spectral Gate (Fullband, -100 dB) → Channel Detection → Normalize (-0 dBFS) → FLAC"
shell_hooks["CommandFlags"] = "8"
shell_hooks["Icon"] = ",".join(("TaskMgr.exe", "-30663"))

shell_hooks.relcd("command")
shell_hooks.make()

shell_hooks[""] = f'"{loc_py}" -m {name_prog} -i "%1" -f -o "%1.flac" -gc 100 -norm -nsr -nbd -nbr -np -v -1'
shell_hooks.relcd(os.path.join(os.pardir, os.pardir)) # exit current command

#---
# exit current command subkey
shell_hooks.relcd(os.path.join(os.pardir, os.pardir))



#---
# exit this commands section
shell_hooks.relcd(os.pardir)


#-=-=-=-#
# Shell commands
shell_commands_items = shell_hooks.as_dict(recursive = False)
shell_commands_items = [i for i in shell_commands_items if i.lower() != "shell"]

shell_hooks.relcd("Shell")

counter = 1
for value in shell_commands_items:
	value_glob = str(counter).rjust(len(str(len(shell_commands_items))) + 1, "0") + "_" + value

	shell_hooks.relcd(value_glob)
	shell_hooks.make()
	shell_hooks[""] = value
	shell_hooks["ExtendedSubCommandsKey"] = os.path.join("*", name_ctxMenus, name_prog, name_commands, value, value + name_globalPipeline)

	if "proc" in value.lower():
		shell_hooks["Icon"] = ",".join(("Shell32.dll", "-16739"))
	else:
		shell_hooks["Icon"] = ",".join(("Shell32.dll", "-16783"))

	shell_hooks.relcd(os.pardir)

	counter += 1

print(f"Re/installed {counter} commands\a")