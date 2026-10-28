"""Manual mass-dimension separation visualization.

Use --model FILE --samples 200 to generate, encode and plot in one command.
Without --model, reads existing embeddings. See docs/PRETRAIN_EVALUATION.md.
Importing this module does not load data, construct a model, or generate samples.
"""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from project.pretraining.visualization import main as _main, prepare_tsne_data, run_tsne_analysis


def main(argv=None):
    return _main("dim", argv)


if __name__ == "__main__":
    raise SystemExit(main())
