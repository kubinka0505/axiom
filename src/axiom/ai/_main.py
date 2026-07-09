from ._core.parser.core import CLI

import warnings
warnings.filterwarnings("ignore", category = ResourceWarning)

#-=-=-=-#

def main():
    args = CLI().parse_args()

    mode = args.mode.lower()

    if mode == "prep":
        from ._core.processors.prep import main as run
    elif mode == "train" or mode == "infer":
        from ._core.processors.model import main as run
    else:
        raise RuntimeError(f"Unknown mode: {mode}")

    run(args)