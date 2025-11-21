Following directory contains simple AI architectures for training sample-rate detection models, which axiom supports.

<a href="https://colab.research.google.com/github/kubinka0505/axiom/blob/master/Files/src/ai/AI.ipynb"><img src="https://shields.io/badge/Colab-Open-F9AB00?&logoColor=F9AB00&style=for-the-badge&logo=Google-Colab" alt="Open in Google Colab"></a>

> [!IMPORTANT]
> Requires [`torch`](pypi.org/project/torch) to work.

> [!IMPORTANT]
> It's early experimental feature. For better AI capabilities, consider using [FLAD](https://github.com/Sg4Dylan/FLAD) or other related software.

> [!NOTE]
> Model consists of 2 architectures. Please visit `model.py` for more information.

```bash
user$os:~ $ axiom -i file.wav -m model.pt ... -v 1
```

## Dataset preparation 📝
Optional.

```bash
user$os:~ $ axiom-ai-prep -i "dataset" -o "dataset_resampled" -k 100
```

## Training 🏋️
```bash
user$os:~ $ axiom-ai-model train -i "dataset_resampled" -o "logs/model.pt" -e 100 -a 2
```

## Inference 🧠
```bash
user$os:~ $ axiom-ai-model infer -i "file.flac" -m "logs/model.pt"
```