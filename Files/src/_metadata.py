import base64

from mutagen import File as mFile
from mutagen.mp3 import MP3
from mutagen.flac import FLAC, Picture
from mutagen.oggvorbis import OggVorbis
from mutagen.id3 import ID3, APIC, USLT, TXXX, WXXX, COMM, ID3NoHeaderError, Frames

from typing import Any, Dict

#-=-=-=-#

MAP_TAG = {
    "DATE": "year",
    "COMMENT": "comment",
    "ORIGINALARTIST": "orig artist",
    "ORIGARTIST": "orig artist",
    "URL": "url",
    "MOOD": "mood",
    "RATING": "rating",
    "LYRICS": "lyrics",
    "ALBUM ARTIST": "album artist",
    "ORIGINALLYRICIST": "orig lyricist",
    "VERSION": "remixed by",
    "RADIOSTATIONNAME": "radio station name",
    "AUDIOFILEURL": "audio file info url",
    "AUTHORURL": "artist url",
    "AUDIOSOURCEURL": "audio src url",
    "RADIOSTATIONURL": "radio station url",
    "BUYCDURL": "buy cd url",
    "LABELNO": "catalog number",
    "INITIALKEY": "initial key"
}

def normalize_tags(filepath: str) -> dict:
    """
    Extracts and normalizes metadata tags and cover art from an audio file.

    Supports MP3, FLAC, and OGG formats. Returns a dictionary of normalized tag names and values, along with cover image data (if present).

    Tag keys are mapped to a consistent internal format using `MAP_TAG` and standard ID3/FLAC/Vorbis fields.

    Args:
        filepath (str): Path to the input audio file.

    Returns:
        tuple:
            - tags (dict): Normalized tag names mapped to their values.
            - cover (dict or None): Dictionary containing embedded image data, or None if no cover art is found.
    """
    tags = {}

    cover = None
    ext = filepath.lower().split(".")[-1]
    audio = mFile(filepath)

    if audio is None:
        return tags, cover

    if ext == "mp3":
        try:
            id3 = ID3(filepath)
        except ID3NoHeaderError:
            return {}, None

        for key in id3.keys():
            frame = id3.getall(key)[0]

            if key.startswith("TXXX"):
                if frame.desc.upper() in MAP_TAG:
                    tags[MAP_TAG[frame.desc.upper()]] = frame.text[0]
            elif key.startswith("WXXX"):
                if frame.desc.upper() in MAP_TAG:
                    tags[MAP_TAG[frame.desc.upper()]] = frame.url
            elif key.startswith("COMM"):
                if frame.desc.upper() in MAP_TAG:
                    tags[MAP_TAG[frame.desc.upper()]] = frame.text[0]
                elif frame.desc == "":
                    tags["comment"] = frame.text[0]
            elif key.startswith("USLT"):
                tags["lyrics"] = frame.text
            elif key in Frames:
                std_map = {
                    "TIT2": "title",
                    "TPE1": "artist",
                    "TALB": "album",
                    "TCON": "genre",
                    "TCOM": "composer",
                    "TCOP": "copyright",
                    "TRCK": "tracknumber",
                    "TPOS": "discnumber",
                    "TPE2": "album artist",
                    "TPE3": "conductor",
                    "TPE4": "remixed by",
                    "TOPE": "orig artist",
                    "TEXT": "lyricist",
                    "TOLY": "orig lyricist",
                    "TENC": "encodedby",
                    "TBPM": "bpm",
                    "TKEY": "initial key",
                    "TSRC": "isrc",
                    "TPUB": "organization",
                    "TCMP": "compilation"
                }

                if key in std_map:
                    tags[std_map[key]] = frame.text[0]

            if isinstance(frame, APIC):
                cover = {
                    "data": frame.data,
                    "mime": frame.mime or "image/jpeg",
                    "desc": frame.desc or "cover"
                }
    elif ext == "flac":
        audio = FLAC(filepath)

        for key, value in audio.tags.items():
            tags[key.lower()] = value[0]

        if audio.pictures:
            pic = audio.pictures[0]
            cover = {
                "data": pic.data,
                "mime": pic.mime,
                "desc": pic.desc or "cover"
            }
    elif ext == "ogg":
        audio = OggVorbis(filepath)

        for key, value in audio.tags.items():
            tags[key.lower()] = value[0]

        if "metadata_block_picture" in audio:
            try:
                raw = base64.b64decode(audio["metadata_block_picture"][0])
                pic = Picture(raw)

                cover = {
                    "data": pic.data,
                    "mime": pic.mime,
                    "desc": pic.desc or "cover"
                }
            except Exception:
                pass

    return tags, cover

def apply_tags(filepath: str, tags: dict, cover: Dict[dict, Any]):
    """
    Applies metadata tags and optional cover art to an audio file.

    Tags must be in the normalized format used internally (e.g., "album artist",
    "original artist", "lyrics", etc.). Function automatically applies the correct
    tagging format depending on the file type (MP3, FLAC, or OGG).

    For MP3, ID3v2.3 tags are written. FLAC and OGG use Vorbis comments.

    Args:
        filepath (str): Path to the target audio file to tag.
        tags (dict): Dictionary of normalized tag names and values.
        cover (dict or None): Dictionary containing cover image data with
            keys: "data" (bytes), "mime" (str), "desc" (str). Set to None to skip.
    """
    ext = filepath.lower().split(".")[-1]

    if ext == "mp3":
        audio = MP3(filepath)

        id3 = ID3(filepath)
        id3.delete()

        for key, value in tags.items():
            if key == "lyrics":
                id3.add(USLT(encoding = 3, lang = "eng", desc = "", text = value))
            elif key == "comment":
                id3.add(COMM(encoding = 3, lang = "eng", desc = "", text = value))
            elif key == "url":
                id3.add(WXXX(encoding = 3, desc = "", url = value))
            elif key in MAP_TAG.values():
                id3.add(TXXX(encoding = 3, desc = key.upper(), text = value))
            else:
                # skip unrecognized
                continue

        if cover:
            id3.add(APIC(
                encoding = 3,
                mime = cover["mime"],
                type = 3,
                desc = cover["desc"],
                data = cover["data"]
            ))

        id3.save(filepath, v2_version = 3)

    elif ext == "flac":
        audio = FLAC(filepath)

        for key, value in tags.items():
            audio[key] = value

        if cover:
            audio.clear_pictures()
            pic = Picture()
            pic.data = cover["data"]
            pic.type = 3
            pic.mime = cover["mime"]
            pic.desc = cover["desc"]
            pic.width = 0
            pic.height = 0
            pic.depth = 0

            audio.add_picture(pic)

        audio.save()

    elif ext == "ogg":
        audio = OggVorbis(filepath)

        for key, value in tags.items():
            audio[key] = value

        if cover:
            pic = Picture()
            pic.data = cover["data"]
            pic.type = 3
            pic.mime = cover["mime"]
            pic.desc = cover["desc"]
            pic.width = 0
            pic.height = 0
            pic.depth = 0

            encoded = base64.b64encode(pic.write()).decode("ascii")
            audio["metadata_block_picture"] = [encoded]

        audio.save()