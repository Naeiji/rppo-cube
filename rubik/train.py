"""Train the recurrent PPO cube agent.

    python -m rubik.train --out my_agent --total-samples 7.5e8

The same command resumes from <out>/checkpoint.pt. Stop with Ctrl-C or SIGTERM: the current iteration
finishes and a checkpoint is written.
"""
from __future__ import annotations

import argparse
from dataclasses import fields

from rubik.ppo import PPOConfig, Trainer


def _bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes", "on")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(PPOConfig):
        kind = _bool if isinstance(f.default, bool) else type(f.default)
        p.add_argument("--" + f.name.replace("_", "-"), type=kind, default=f.default)
    Trainer(PPOConfig(**vars(p.parse_args(argv)))).train()


if __name__ == "__main__":
    main()
