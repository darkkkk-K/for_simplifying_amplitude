"""Pretrain the RoPE encoder with the configured contrastive objective."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/pretrain.json')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    print(json.dumps(config, ensure_ascii=False, indent=2))
    if args.run:
        import os
        import torch
        from training.pretrain import train_model
        if (ROOT/config['output_dir']).exists():
            parser.error('Choose a new output_dir for this training run.')
        config['use_amp'] = bool(config['use_amp'] and torch.cuda.is_available())
        os.chdir(ROOT)
        train_model(config, if_generate_data=False)


if __name__ == '__main__':
    main()
