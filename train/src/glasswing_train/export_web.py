"""Convert trained f32 weights to f16 for the browser: half the download, same file format.

    uv run python -m glasswing_train.export_web model.safetensors model-f16.safetensors

Weights of a trained model sit well inside f16's range (±65504); what's lost is precision
(~3 significant digits instead of ~7). The Rust engine widens them back to f32 on load.
"""

from __future__ import annotations

import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    with safe_open(str(src), framework="pt") as f:
        meta = f.metadata()
        tensors = {k: f.get_tensor(k).half().contiguous() for k in f.keys()}
    worst = max(t.float().abs().max().item() for t in tensors.values())
    assert worst < 60000, f"a weight of {worst} would overflow f16"
    save_file(tensors, str(dst), metadata=meta)
    print(
        f"{src.stat().st_size / 1e6:.1f} MB -> {dst.stat().st_size / 1e6:.1f} MB "
        f"(largest |weight| {worst:.2f})"
    )


if __name__ == "__main__":
    main()
