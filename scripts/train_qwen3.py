from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from gpt2.train import main as gpt2_main

    gpt2_main()


if __name__ == "__main__":
    main()
