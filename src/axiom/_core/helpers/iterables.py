from bisect import bisect_left, bisect_right

def snap(value: float, options: list, left: bool = False) -> iter:
	"""
	Snaps a value to the next closest value in a sorted list of options.

	If the value is greater than all options, the last option is returned.

	Parameters
	----------
		value (float):
			The target value to be snapped.

		left (bool):
			Determines whether values are snapped to left or right element of an iterable.

		options (list):
			A list of numeric values to snap to.

	Returns
	-------
		numeric:
			The closest greater than or equal value in the list, or the maximum if none are greater.

	Example
	-------
		>>> snap(200, [128, 192, 256])
		256
	"""
	func = bisect_left if left else bisect_right

	opts = sorted(options)
	idx = func(opts, value)

	if idx >= len(opts):
		return opts[-1]

	return opts[idx]