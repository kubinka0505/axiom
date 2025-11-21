import os
from shutil import rmtree
from setuptools import setup, find_packages

# Remove directories after installing
cleanup = True

#-=-=-=-#

__title__   = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
__author__  = "kubinka0505"
__credits__ = __author__

#-=-=-=-#

# Cleanup
if cleanup:
	if "install" in os.sys.argv:

		# Built directories
		dirs = [
			"build", "dist", "name",
			f"{__title__}.egg-info"
		]

		for directory in dirs:
			rmtree(directory, True)

directory = "src"
if os.path.exists(directory):
	os.rename(directory, __title__)

#-=-=-=-#

tags = [__title__,
	"audio", "samplerate", "sample rate", "sampling rate", "cutoff", "channel", "channels", "peak",
	"audio estimation", "samplerate estimation", "sample rate estimation", "sampling rate estimation", "cutoff estimation", "channel estimation", "channels estimation", "peak estimation",
	"audio detection", "samplerate detection", "sample rate detection", "sampling rate detection", "cutoff detection", "channel detection", "channels detection", "peak detection",
]

_project_url = f"https://github.com/{__author__}/{__title__}"

setup(
	name = __title__,
	description = "Audio files parameters estimation.",
	version = "1.0",
	author = __author__,
	maintainer = __credits__,
	url = _project_url,
	keywords = tags,
	classifiers = [
		"Development Status :: 5 - Production/Stable",
		"Environment :: Console",
		"Intended Audience :: Developers",
		"Intended Audience :: End Users/Desktop",
		"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
		"Natural Language :: English",
		"Operating System :: OS Independent",
		"Programming Language :: Python :: 3 :: Only",
		"Topic :: Multimedia :: Sound/Audio",
		"Topic :: Multimedia :: Sound/Audio :: Analysis",
		"Typing :: Typed"
	],
	python_requires = ">= 3.9",
	project_urls = {
		"GitHub": _project_url,
		"Homepage": _project_url,
		"Issue tracker": _project_url + "/issues",
	},
	install_requires = [
		"numpy >= 1.26.4",
		"scipy >= 1.12.0",
		"librosa >= 0.10.2.post1",
		"soundfile >= 0.12.1",
		"pytimeparse >= 1.1.8",
		"mutagen >= 1.47.0",
		"colorama >= 0.4.6",
		"pydub >= 0.25.1",
		# "torch",
	],
	entry_points = {
		"console_scripts": [
			"axiom = axiom.cli_core:main",
			"axiom-ai-prep = axiom.ai.prep:main",
			"axiom-ai-model = axiom.ai.model:main"
		]
	},
	packages = find_packages()
)

#-=-=-=-#

if os.path.exists(__title__):
	os.rename(__title__, directory)