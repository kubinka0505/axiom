import re
from numbers import Real

def clamp(number: int | float, minimum: int | float, maximum: int | float) -> int | float:
	"""
	Clamps the number to desired range.

	Parameters
	----------
		number (int | float):
			The clamped number.

		minimum (int | float):
			The minimum value to clamp `number` to.

		maximum (int | float):
			The maximum value to clamp `number` to.

	Returns
	-------
		int | float:
			The clamped number.
	"""
	return min(max(minimum, number), maximum)

def percentage(percent: float, whole: float, rounding: int = 10) -> float:
	"""
	Calculates the percentage value of a part over a whole.

	Parameters
	----------
		percent (float):
			The part value.

		whole (float):
			The total or whole value.

		rounding (int, optional):
			Number of decimal places to round the result.

	Returns
	-------
		float:
			The calculated percentage.

	Example
	-------
		>>> percentage(25, 200)
		12.5
	"""
	return round((percent * 100) / whole, rounding)

def to_readable(value: str | Real) -> int | float | None:
	"""
	Converts numeric-ish strings into numbers.

	Examples
	--------
		- "2e2" -> 200
		- "22.01k" -> 22010
		- "5" -> 5
		- "2.5" -> 2.5
		- "1.25k" -> 1250
	"""
	unitmap = {"k": 3, "m": 6, "g": 9}

	def optimize(number: float) -> int | float:
		return int(number) if number.is_integer() else number

	if isinstance(value, Real):
		return optimize(float(value))

	value = str(value).strip().lower()

	# plain or scientific notation
	try:
		number = float(value)
		return optimize(number)
	except ValueError:
		pass

	# suffix units
	match = re.fullmatch(
		rf"([0-9]*\.?[0-9]+)([{''.join(unitmap)}])",
		value,
	)

	if match:
		number, suffix = match.groups()
		result = float(number) * (10 ** unitmap[suffix])
		return optimize(result)

	return None