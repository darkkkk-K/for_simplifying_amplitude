"""Manual term clustering within expression groups encoder evaluation.

See docs/PRETRAIN_EVALUATION.md. Default phase is plan; generation/encoding require explicit --phase.
Importing this module does not load data, construct a model, or generate samples.
"""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from project.pretraining.evaluation import main as _main


def main(argv=None):
    return _main("sim", argv)


if __name__ == "__main__":
    raise SystemExit(main())
