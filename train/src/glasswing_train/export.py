"""Save and load models as safetensors: a JSON header plus raw little-endian tensors.

The config travels in the header's `__metadata__`, so one file is enough to rebuild the model.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from glasswing_train.model import ModelConfig, Transformer

FORMAT = "glasswing-model-v1"


def save_model(model: Transformer, path: Path) -> None:
    tensors = {
        k: v.detach().to("cpu", torch.float32).contiguous() for k, v in model.state_dict().items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={"format": FORMAT, "config": model.cfg.to_json()})


def load_model(path: Path, device: str = "cpu") -> Transformer:
    with safe_open(str(path), framework="pt", device=device) as f:
        meta = f.metadata()
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} file")
        model = Transformer(ModelConfig.from_json(meta["config"]))
        model.load_state_dict({k: f.get_tensor(k) for k in f.keys()}, strict=True)
    return model.to(device)
