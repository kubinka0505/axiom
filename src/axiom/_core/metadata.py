import base64
from contextlib import suppress

from typing import Any, Dict, Tuple, Optional

from mutagen.wave import WAVE
from mutagen import File as mFile
from mutagen.flac import FLAC, Picture
from mutagen.oggvorbis import OggVorbis

import mutagen.id3 as _core_id3
from mutagen.id3 import (
	ID3, ID3NoHeaderError,

	TIT2, TPE1, TALB, TCON, TDRC, TPE2,
	TCOM, TPE3, TPE4, TRCK, TPOS,
	TEXT, TBPM, TCOP, TPUB, TENC,
	TKEY, TSRC, TOPE, TOLY,

	USLT, COMM, WXXX, APIC, TXXX
)

# -----------------------------
# NORMALIZATION MAP
# -----------------------------

TAG_MAP = {
	"title":              ["TIT2", "TITLE"],
	"artist":             ["TPE1", "ARTIST"],
	"album":              ["TALB", "ALBUM"],
	"genre":              ["TCON", "GENRE"],
	"year":               ["TDRC", "DATE", "YEAR"],
	"album artist":       ["TPE2", "ALBUMARTIST", "ALBUM ARTIST"],
	"composer":           ["TCOM", "COMPOSER"],
	"conductor":          ["TPE3", "CONDUCTOR"],
	"remixed by":         ["TPE4", "REMIXEDBY", "VERSION"],
	"tracknumber":        ["TRCK", "TRACKNUMBER"],
	"discnumber":         ["TPOS", "DISCNUMBER"],
	"lyricist":           ["TEXT", "LYRICIST"],
	"bpm":                ["TBPM", "BPM"],
	"copyright":          ["TCOP", "COPYRIGHT"],
	"organization":       ["TPUB", "ORGANIZATION"],
	"encodedby":          ["TENC", "ENCODEDBY"],
	"initial key":        ["TKEY", "INITIALKEY"],
	"isrc":               ["TSRC", "ISRC"],
	"orig artist":        ["TOPE", "ORIGINALARTIST", "ORIGARTIST"],
	"orig lyricist":      ["TOLY", "ORIGINALLYRICIST"],
	"comment":            [None,   "COMMENT"],
	"lyrics":             [None,   "LYRICS"],
	"url":                [None,   "URL"],
	"mood":               [None,   "MOOD"],
	"rating":             [None,   "RATING"],
	"catalog number":     [None,   "LABELNO"],
}


# "TIT2" -> "title"
ID3_FRAME_TO_KEY = {v[0]: k for k, v in TAG_MAP.items() if v[0]}

# "title" -> TIT2 (class)
ID3_KEY_TO_FRAME = {
    k: getattr(_core_id3, v[0])
    for k, v in TAG_MAP.items() if v[0]
}

# "DATE" -> "year",  "ALBUMARTIST" -> "album artist", etc.
ALIAS_TO_KEY = {alias: k for k, v in TAG_MAP.items() for alias in v[1:]}

# ============================================================
# READ / NORMALIZE
# ============================================================

def normalize_tags(filepath: str) -> Tuple[dict, Optional[dict]]:
	tags = {}
	cover = None
	ext = filepath.lower().split(".")[-1]

	audio = mFile(filepath)
	if audio is None:
		return {}, None

	# ---------------- MP3 / WAV (ID3) ----------------
	if ext in ("mp3", "wav"):
		try:
			if ext == "mp3":
				id3 = ID3(filepath)
			else:
				audio = WAVE(filepath)

				if audio.tags is None:
					return {}, None

				id3 = audio.tags
		except ID3NoHeaderError:
			return {}, None

		for key, frame in id3.items():
			frame_name = key[:4]

			if frame_name in ID3_FRAME_TO_KEY:
				val = getattr(frame, "text", None)

				if val:
					tags[ID3_FRAME_TO_KEY[frame_name]] = val[0]

			elif key.startswith("COMM"):
				tags["comment"] = frame.text[0]

			elif key.startswith("USLT"):
				tags["lyrics"] = frame.text

			elif key.startswith("TXXX"):
				normalized = ALIAS_TO_KEY.get(frame.desc.upper(), frame.desc.lower())
				tags[normalized] = frame.text[0]

			elif key.startswith("WXXX"):
				tags["url"] = frame.url

			elif isinstance(frame, APIC):
				cover = {
					"data": frame.data,
					"mime": frame.mime,
					"desc": frame.desc or "cover"
				}

	# ---------------- FLAC ----------------
	elif ext == "flac":
		audio = FLAC(filepath)

		for k, v in audio.tags.items():
			normalized = ALIAS_TO_KEY.get(k.upper(), k.lower())
			tags[normalized] = v[0]

		if audio.pictures:
			pic = audio.pictures[0]
			cover = {
				"data": pic.data,
				"mime": pic.mime,
				"desc": pic.desc or "cover"
			}

	# ---------------- OGG ----------------
	elif ext == "ogg":
		audio = OggVorbis(filepath)

		for k, v in audio.tags.items():
			normalized = ALIAS_TO_KEY.get(k.upper(), k.lower())
			tags[normalized] = v[0]

		if "metadata_block_picture" in audio:
			with suppress():
				raw = base64.b64decode(audio["metadata_block_picture"][0])
				pic = Picture(raw)

				cover = {
					"data": pic.data,
					"mime": pic.mime,
					"desc": pic.desc or "cover"
				}

	return tags, cover

# ============================================================
# WRITE
# ============================================================

def apply_tags(filepath: str, tags: dict, cover: Dict[str, Any] = None):
	ext = filepath.lower().split(".")[-1]

	# ---------------- MP3 ----------------
	if ext == "mp3":
		try:
			id3 = ID3(filepath)
		except ID3NoHeaderError:
			id3 = ID3()

		id3.delete()
		id3 = ID3()

		for k, v in tags.items():
			if k in ID3_KEY_TO_FRAME:
				id3.add(ID3_KEY_TO_FRAME[k](encoding = 3, text = [str(v)]))

			elif k == "lyrics":
				id3.add(USLT(encoding = 3, lang = "eng", desc = "", text = str(v)))

			elif k == "comment":
				id3.add(COMM(encoding = 3, lang = "eng", desc = "", text = str(v)))

			elif k == "url":
				id3.add(WXXX(encoding = 3, desc = "", url = str(v)))

			else:
				id3.add(TXXX(encoding = 3, desc = k, text = [str(v)]))

		if cover:
			id3.add(APIC(
				encoding = 3,
				mime = cover["mime"],
				type = 3,
				desc = cover.get("desc", "cover"),
				data = cover["data"]
			))

		id3.save(filepath, v2_version = 3)

	# ---------------- FLAC ----------------
	elif ext == "flac":
		audio = FLAC(filepath)
		audio.clear()

		for k, v in tags.items():
			audio[k.upper()] = str(v)

		if cover:
			audio.clear_pictures()

			pic = Picture()
			pic.data = cover["data"]
			pic.mime = cover["mime"]
			pic.type = 3
			pic.desc = cover.get("desc", "cover")

			audio.add_picture(pic)

		audio.save()

	# ---------------- OGG ----------------
	elif ext == "ogg":
		audio = OggVorbis(filepath)
		audio.clear()

		for k, v in tags.items():
			audio[k.upper()] = str(v)

		if cover:
			pic = Picture()
			pic.data = cover["data"]
			pic.mime = cover["mime"]
			pic.type = 3
			pic.desc = cover.get("desc", "cover")

			encoded = base64.b64encode(pic.write()).decode("ascii")
			audio["metadata_block_picture"] = [encoded]

		audio.save()

	# ---------------- WAV (ID3) ----------------
	elif ext == "wav":
		audio = WAVE(filepath)

		if audio.tags is None:
			audio.add_tags()
		else:
			audio.tags.delete(filepath)
			audio.add_tags()

		id3 = audio.tags

		for k, v in tags.items():
			if k in ID3_KEY_TO_FRAME:
				id3.add(ID3_KEY_TO_FRAME[k](encoding = 3, text = [str(v)]))

			elif k == "lyrics":
				id3.add(USLT(encoding = 3, lang = "eng", desc = "", text = str(v)))

			elif k == "comment":
				id3.add(COMM(encoding = 3, lang = "eng", desc = "", text = str(v)))

			elif k == "url":
				id3.add(WXXX(encoding = 3, desc = "", url = str(v)))

			else:
				id3.add(TXXX(encoding = 3, desc = k, text = [str(v)]))

		if cover:
			id3.add(APIC(
				encoding = 3,
				mime = cover["mime"],
				type = 3,
				desc = cover.get("desc", "cover"),
				data = cover["data"]
			))

		audio.save()