<details closed>
    <summary>But I can use my eyes to look at the spectrogram and then determine the cutoff, so why would I use this?</summary>

Because axiom is limiting manual analysis steps to absolute minimum. You just have to provide files, which is simpler than looking on a time/frequency graph, which spectrogram is. Instead of looking, you have to either type OR drag and drop a file or a directory (depending on terminal support).
</details>

<details closed>
	<summary>Is it lossless audio checker?</summary>

**No.** The most high-sounding term that could be used could be “transcoding detector”.
</details>

<details closed>
	<summary>Why would I want to estimate channels?</summary>

Despite axiom not being very suitable for short signals due to large probability of errors occurrences — and therefore inaccuracy — this one was made for music producers. Even to this day, sample packs creators often tend to export files in mono and then in stereo, for some reason (compatibility is not one of them). I wanted to give them the tool to estimate this and save disk space.
</details>

<!--details closed>
	<summary>Have you considered bit-depth estimation?</summary>

It's not implementable due to requirement of both used dithering algorithm and a relative signal. Furthermore, differences between them are too small to be measurable.
</details-->

<details closed>
	<summary>Does it support surround files in 5.1 format or above?</summary>

Channels functionality is strictly limited to mono due to processing time optimization. The only estimators that don't convert the signal to mono are `peak` and obviously `channels` ones.
</details>

<details closed>
	<summary>Why is the signal extended? Doesn't decrease accuracy?</summary>

It may do, but it's a small price for implementation of a universal yet safe solution for short audio signals that aren't processable in their normal fashion due to FFT window size.
</details>

<details closed>
	<summary>What does perceptual difference means?</summary>

It shows the deviation from human hearing range and estimated sample rate. Always 0 if estimated sample rate > 40000.
</details>

<details closed>
	<summary>How is FLAC bitrate estimated? Is that AI?</summary>

No, but by the most unethical way — remove all input file metadata, apply maximum compression level and return the bitrate of an output file stream.
</details>

<details closed>
	<summary>How have you exported the animated graph image?</summary>

I have created `_AXIOM_FRAMES` directory inside current working directory and ran the program with `--verbosity` flag set to `2` without `--model` argument.

Steps amount (`STEP_CLAMP_VALUE`) was changed by trial and error method to `9600` inside the code.
</details>

<details closed>
	<summary>Why does the estimated sample rate turns green even if it's lower than the original one?</summary>

Because I treat postprocessal upsampling as ethical if perceptual difference is none. If it's any different number, it's changed to red immediately.
</details>