Following directory contains simple AI architecture for training sample-rate detection model.

<a href="https://colab.research.google.com/github/kubinka0505/axiom/blob/master/docs/AI.ipynb"><img src="https://shields.io/badge/Colab-Open-F9AB00?&logoColor=F9AB00&style=for-the-badge&logo=Google-Colab" alt="Open in Google Colab"></a>

> [!IMPORTANT]
> Requires [`torch`](pypi.org/project/torch) to work.

> [!IMPORTANT]
> It's early experimental feature.
> 
> For better AI capabilities, consider using spectral-image based [FLAD](https://github.com/Sg4Dylan/FLAD) or other related software.

```bash
user$os:~ $ axiom -i file.wav -m model.pt ... -v 1
```

## Dataset preparation 📝
Optional.

```bash
user$os:~ $ axiom-ai prep -i "dataset" -o "dataset_resampled" -k 100
```

## Training 🏋️
```bash
user$os:~ $ axiom-ai train -i "dataset_resampled" -o "logs/model.pt" -e 100
```

## Inference 🧠
```bash
user$os:~ $ axiom-ai infer -i "file.flac" -m "logs/model.pt"
```