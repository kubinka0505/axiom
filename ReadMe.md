<img src="https://raw.githubusercontent.com/kubinka0505/axiom/refs/heads/master/Documents/Pictures/Logo.svg" width=150>

<a href="https://github.com/kubinka0505/axiom/commit"><img src="https://custom-icon-badges.demolab.com/github/last-commit/kubinka0505/axiom?logo=commit&style=for-the-badge&" alt="Last commit date"></a>　<a href="https://github.com/kubinka0505/axiom/blob/master/License.txt"><img src="https://custom-icon-badges.demolab.com/github/license/kubinka0505/axiom?logo=law&color=red&style=for-the-badge&" alt="View license"></a>　<a href="https://app.codacy.com/gh/kubinka0505/axiom"><img src="https://img.shields.io/codacy/grade/23c2a78e2c98400bb880c57358395955?logo=codacy&style=for-the-badge" alt="View grade"></a>　<a href="https://colab.research.google.com/github/kubinka0505/axiom/blob/master/Documents/Notebook.ipynb"><img src="https://shields.io/badge/Colab-Open-F9AB00?&logoColor=F9AB00&style=for-the-badge&logo=Google-Colab" alt="Open in Google Colab"></a>

## Description 📝
Placebo for audiophiles, the successor of [soundless](https://github.com/kubinka0505/soundless) and one of a kind software.

<img src="https://raw.githubusercontent.com/kubinka0505/axiom/refs/heads/master/Documents/Pictures/Spectrograms/Graph.gif" width=350>

### Features ✨
- Sample rate estimation
  - No AI usage [by default](Files/src/ai)
- Bit depth estimation
- Channel amount estimation
- Bitrate calculation
- dBFS peak estimation

> [!NOTE]
> This software can work without `pydub` but it offers extended capabilities.

> [!NOTE]
> For more explanations, please consider visiting [frequently asked questions](wiki/FAQ) wiki page.

### Genesis 🏛️
April 21, 1997, can be considered as the worst day in the history of music quality. The release of [Winamp](https://wikipedia.org/wiki/Winamp) and the rapid rise of the [MP3 format](https://wikipedia.org/wiki/MP3) that came from it began a revolution in accessibility, but it came at a cost: music quality dropped. Lossless releases were encoded to questionable quality by untrained people. Fullband sound went to further plan, and when it resurfaced, it often came with degraded dynamics and equivalent quality. Sometimes worse.

Since 2018 a flood of so-called “sound engineers” trained by YouTube, unaware of even the basics of audio science, has filled — or rather attempted to redefine — the industry with poor recordings and narrowband samples. The true craft of music production has been buried under noise.

This is why I built `axiom` — not to “improve” the sound<sup>1</sup>, but to expose what it **might** be.

> What is the **approximate** frequency cutoff?

> How many channels are actually there?

> Is this high‑res file **really** what it **claims** to be?

<sup>1</sup> - although [it **MAY** upsample](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html) [fake-lossless](https://raw.githubusercontent.com/kubinka0505/axiom/refs/heads/master/Documents/Pictures/Spectrograms/Fake.png) files a bit, so I apologize.

### Roadmap 🏁
- [ ] Bug fixing 🐛
- [ ] Writing to multiple files 🔢
- [x] Cross-extension metadata copying 🔀

## Installation 🖥️
1. [`git`](https://git-scm.com) (recommended)
```bash
git clone https://github.com/kubinka0505/axiom
cd axiom/Files
python setup.py install
```

2. [`pip`](https://pypi.org/project/pip)
```bash
python -m pip install "git+https://github.com/kubinka0505/axiom#egg=axiom&subdirectory=Files" -U --use-deprecated=legacy-resolver
```

## Usage 📝

### Module (simple)
```python
>>> from axiom import Axiom
>>> 
>>> # Create object
>>> axiom_obj = Axiom("HD.flac")
>>> 
>>> # Estimate sample rate
>>> axiom_obj.sample_rate()
{'HD.flac': {'samplerate': 33462, 'cutoff': 16731.298828125}}
>>> 
>>> # Estimate bit depth
>>> axiom_obj.bit_depth()
{'HD.flac': {'bit depth': 10, 'bit depth snapped': 16}
>>> 
>>> # Estimate channels
>>> axiom_obj.channels()
{'HD.flac': {'channels': 2}
>>> 
>>> # Estimate peak
>>> axiom_obj.peak()
{'HD.flac': {'peak': 0.0}
>>> 
>>> # All in one
>>> axiom_obj.aio()
{'HD.flac': {'samplerate': 33462, 'cutoff': 16731.298828125, 'bit depth': 10, 'bit depth snapped': 16, 'channels': 2, 'bitrate': 1070803.125, 'peak': 0.0}}
```

### Module (advanced)
```python
>>> from axiom import Axiom
>>> 
>>> # Files
>>> to_process = [
...     "%UserProfile%/file.mp3",
...     "~/audio.wav",
...     "../song.flac",
...     "sound.ogg",
...     "../files"
... ]
... 
>>> axiom_obj = Axiom(to_process, recursive = True)
>>> 
>>> results = axiom_obj.estimate(
...     start = 10,       # Begin from frame 10
...     duration = "1s",  # Analyze (max) 1 second of file
...     freq_step = 200   # Frequency drop in sample estimation
... )
```

### Command-line console script

<details open>
	<summary><b>Simple display</b> 🧐</summary>
    
```bash
user$os:~ $ # All in one, separated
user$os:~ $ axiom -i "file.mp3"
14384 7192.08984375 14 16 1 -0.001
```
</details>

<details open>
	<summary><b>Quiet convert</b> 🔕</summary>
    
```bash
user$os:~ $ axiom -i "file.mp3" -o "file_resampled.wav" -v -1
user$os:~
```
</details>

<details open>
	<summary><b>Advanced</b> ⚙️</summary>
    
```bash
user$os:~ $ # Start analysis from 2 seconds (-ss 2.5s) for 25000 samples (-to 25000)
user$os:~ $ # with recursive processing (-r) and verbose output (-v 2)
user$os:~ $ axiom -i "../dir" -ss 2s -to 25000 -f -r -v 1
[INFO] Loaded file:                     file_18400.wav
[INFO] Information:                     44100 Hz, 28680 samples (00:00.650), Mono, 56.09 KB
[INFO] ----------
[INFO] Processing samples:              0 -> 25000 (each 1) (00:00.566893) (87.17%)
[INFO] Estimated sample rate:           18841 Hz (-25259 Hz) (Linear -57.277%) (Perceptual -7.602%)
[INFO] Estimated bit depth:             10 bits (Normalized to 16 bits for safety)
[INFO] Estimated channels amount:       1 (Original: 1)
[INFO] Calculated bitrate:              301.456 kb/s
[INFO] Estimated peak:                  -0.091 dBFS
[INFO] ----------
[INFO] Loaded file:                     high_quality.flac
[INFO] Information:                     96000 Hz, 47217826 samples (08:11.852), Stereo, 88.74 MB
[INFO] ----------
[INFO] Processing samples:              192000 -> 217000 (each 1) (00:00.260417) (0.46%)
[INFO] Estimated sample rate:           48187 Hz (-47813 Hz) (Linear -49.805%) (Perceptual -0.0%)
[INFO] Estimated bit depth:             17 bits (Normalized to 32 bits for safety)
[INFO] Estimated channels amount:       2 (Original: 2)
[INFO] Calculated bitrate:              1541.984 kb/s
[INFO] Estimated peak:                  -11.207 dBFS
[INFO] ----------
[INFO] Processed 2 files in 00:02.772.
```
</details>

# Disclaimer ⚠️
This software:
   - Was not designed to estimate parameters of short signals.
      - Subjective measurements of FFT values related to sample rate estimation were not performed.
   - Uses **only** vertical scanning in sample rate estimation.
   - May return invalid sample rates estimations for signals with very low sample rate.
   - **Will** return invalid bit depth estimations.
   - Does not denoise files before processing.