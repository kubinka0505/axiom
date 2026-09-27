## 🤔 General

<details closed>
	<summary>I can use my eyes to look at the spectrogram and then approximately determine the sample rate, so why would I use this?</summary>

Because attempt of this module is to limit the manual analysis steps to absolute minimum. You just have to provide files, which is — I dare say — simpler than the deep analysis of a time/frequency graph, which spectrogram is.
</details>

<details closed>
	<summary>Is it lossless audio checker?</summary>

<b>No.</b>

The most high-sounding term that could be used would be “transcoding detector”.
</details>

<details closed>
	<summary>Why would I want to estimate channels?</summary>

Despite axiom not being very suitable tool for especially short signals due to large probability of subjective errors occurrences — and therefore inaccuracy — this one was made targeting music producers.

Popular sample packs creators often release their content with monophonic files simply disguised as stereo, the reason of which (compatibility is NOT one of them) is unknown to me.

I wanted to give them the tool to estimate this and save disk space.
</details>

<details closed>
	<summary>Why my output file is denoised above 20 kHz? It looks ugly.</summary>

It all happens because of spectral gate.
You can disable it by setting `-gc` argument to `inf,inf`. (`-gc "inf, inf"`)

<details closed>
	<summary>ℹ️ More about spectral gate</summary>

The spectral (noise) gate is an unconventional, technically low-level yet innovative solution in the audio file parameters estimation field.

Nominally, its advanced versions are being used in complex algorithms, while the version included in this software has been configured for files that are:
- lossy
- with a relatively low dynamic range (5–8 dBFS)
- and with “unreliable” sample rates.

<img src="https://raw.githubusercontent.com/kubinka0505/axiom/refs/heads/master/docs/img/Spectrograms/Fake.png" width=200>

For example, applying a spectral gate above 20 kHz to a loud MP3 file at the highest quality allowed by its encoder CAN make the sample rate estimation more reliable.
</details>
</details>



## 📜 Ethical

<details closed>
	<summary>Why is the signal extended? Doesn't decrease accuracy?</summary>

It's a small price for implementing an universal yet safe solution for audio signals with ANY length, that aren't processable in their normal fashion due to FFT window size.
</details>

<details closed>
	<summary>Why does the estimated sample rate turns green even if it's lower than the original one?</summary>

Because I treat postprocessal upsampling as ethical ONLY if the perceptual difference is none.

If it's any different number, it's changed to red immediately.

Minority of people listen to downsampled music. Even less expect its quality to be inherited.
</details>



## ⚙️ Technical

<details closed>
	<summary>What does perceptual difference means?</summary>

It shows the deviation from human hearing range and estimated sample rate. Always 0 if estimated sample rate > 40000.

$\frac{20000}{2} \equal 20000$
</details>

<details closed>
	<summary>How is FLAC bitrate estimated?</summary>

By the most unethical way — remove all input file metadata, apply maximum compression level and return the bitrate of an output file stream.
</details>

<details closed>
	<summary>Does it support surround files in 5.1 format or above?</summary>

Channels functionality is strictly limited to mono due to processing time optimization.

The only estimators that don't convert the signal to mono are `peak` and obviously `channels` ones.
</details>

<!--details closed>
	<summary>How have you exported the animated graph image?</summary>

I have created `_AXIOM_FRAMES` directory inside current working directory and ran the program with `--verbosity` flag set to `2` without `--model` argument.

Steps amount (`STEP_CLAMP_VALUE`) was changed by trial and error method to `9600` inside the code.
</details-->

<details closed>
	<summary>Why estimated bit depth of input file is so low?</summary>

The estimated bit depth reflects the signal's effective quantization resolution rather than its stored sample format.

Signals with a limited dynamic range occupy fewer distinguishable quantization levels, so they can be represented with fewer effective bits.

Consequently, quiet or low-amplitude signals—even when stored as 32-bit or 64-bit floating-point samples—may produce low estimated bit depths.

Likewise, applying bit crushing reduces the effective resolution and leads to similar estimates.
</details>

<details closed>
	<summary>AI model is unreliable and returns invalid results.</summary>

I dare say that the fastest solution to that issue is using your eyes.

Unfortunately, I have absolutely no relevant experience in the field of building artificial intelligence architectures or/and models, so I can't do anything about it sadly.

I appreciate any helpful suggestions and feedback, though.
</details>