import logging

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