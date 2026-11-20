"""Run the shortest-path algorithm-selection study."""
from __future__ import annotations

import argparse
from pathlib import Path

from algoselect.study import execute


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shortest-path algorithm selection study")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    print("Benchmarking shortest-path algorithms and fitting selectors...")
    path = execute(args.out, seed=args.seed)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
