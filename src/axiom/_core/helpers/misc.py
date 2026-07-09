def hex2ansi(hexcode: str, fore: bool = True) -> str:
	"""
	Return ANSI escape code for 24-bit color from a hex code string.

	Parameters
	----------
		hexcode (str):
			Hex color code, e.g. "fc0", "#ffcc00", "F6A".

		fore (bool):
			If True, produce a foreground code (38;2); otherwise background (48;2).

	Returns
	-------
		str:
			ANSI escape sequence for that color.
	"""
	hexcode = hexcode.strip().lstrip("#")

	# expand shorthand like "FC0" -> "FFCC00"
	if len(hexcode) == 3:
		hexcode = "".join(c * 2 for c in hexcode)

	if len(hexcode) != 6:
		raise ValueError('Hex code must be 3 or 6 hex characters (after stripping "#").')

	try:
		r = int(hexcode[0:2], 16)
		g = int(hexcode[2:4], 16)
		b = int(hexcode[4:6], 16)
	except ValueError as e:
		raise ValueError(f'Invalid hex color "{hexcode}"') from e

	prefix = "38" if fore else "48"
	return f"\033[{prefix};2;{r};{g};{b}m"