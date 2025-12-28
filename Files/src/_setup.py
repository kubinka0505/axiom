import logging
from colorama import Fore, Back, init, just_fix_windows_console

from ._helper import hex2ansi

init(autoreset = True)
just_fix_windows_console()

#-=-=-=-#
# Variables

DEFAULT_VALUE_FFT = 1024

STEP_CLAMP_VALUE = 800 # higher the better

TOLERANCE_SUBJECTIVE_SOFT = 1000
TOLERANCE_SUBJECTIVE_HARD = 1000
MAX_DURATION_DISPLAY = "9E7"

SAMPLE_LARGE = "1E7"
SAMPLE_MIN_EXTEND = 1E5

NORMALIZE_MIN = -100
NORMALIZE_MAX = 0

EXTENSIONS_DISPLAY = {
	"Wave File": "*.wav",
	"MPEG Audio Layer 3": "*.mp3", # produces weird artifacts when saving
	"Free Lossless Audio Codec": "*.flac",
	"Ogg Vorbis": "*.ogg"
}

#-=-=-=-#
# Variables | post-processing
# Do not modify!

EXTENSIONS_VALID = {ext.strip("*.").lower() for ext in EXTENSIONS_DISPLAY.values()}

MAX_DURATION = int(float(MAX_DURATION_DISPLAY))
SAMPLE_LARGE = int(float(SAMPLE_LARGE))

NORMALIZE_MIN = max(-100, NORMALIZE_MIN)
NORMALIZE_MAX = min(NORMALIZE_MAX, 0)

RATES_MP3_SAMPLE = 32000, 44100, 48000

RATES_MP3_BIT = range(32, 320 + 1)
RATES_MP3_BIT = [n * 1000 for n in RATES_MP3_BIT]

RATES_OGG_BIT = range(32, 500 + 1)
RATES_OGG_BIT = [n * 1000 for n in RATES_OGG_BIT]

DEPTHS_BIT = 16, 32
DEPTHS_BIT_FLAC = 16, 24, 32

# technically from 4 to 32 but compatibility may be narrow accross soft/hardware
# DEPTHS_BIT_FLAC = [f for f in range(4, 32 + 1)]

#-=-=-=-#
# Colors

colors_fore = {
	"Gold": "FC0",
	"GoldWhite": "FD7",
	"Red": "C10",
	"Pink": "F6A",
	"RedDark": "900",
	"Gray": "999",
	"Grey": "999",
	"Lime": "0C0",
	"Orange": "F90",
	"Magenta": "C4F",
	"Blue": "4AF"
}

colors_back = {
	"Highlight": "222",
	"Panel": "0F1",
	"GoldDark": "430"
}

# Helpers
Fore.hex2ansi = staticmethod(lambda h: hex2ansi(h, fore = True))
Back.hex2ansi = staticmethod(lambda h: hex2ansi(h, fore = False))

# Dynamically add attributes
for name, code in colors_fore.items():
	setattr(Fore, name.upper(), Fore.hex2ansi(code))

for name, code in colors_back.items():
	setattr(Back, name.upper(), Back.hex2ansi(code))

#-=-=-=-#
# Logger

colors_loglevel = {
	"DEBUG": Fore.GRAY,
	"INFO": Fore.BLUE,
	"WARNING": Fore.GOLD,
	"ERROR": Fore.RED,
	"CRITICAL": Fore.REDDARK,
}

# Custom formatter that colors levelname
class ColorFormatter(logging.Formatter):
	def format(self, record):
		color = colors_loglevel.get(record.levelname, "")
		record.levelname = Fore.RESET + color + record.levelname + Fore.RESET

		return super().format(record)

# Configure logger
logger = logging.getLogger("axiom")
logger.setLevel(logging.NOTSET)

ch = logging.StreamHandler()
ch.setFormatter(ColorFormatter("[%(levelname)s] %(message)s"))

logger.addHandler(ch)