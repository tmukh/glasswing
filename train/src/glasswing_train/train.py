"""Train the model on data/train.bin (see data.py).

    uv run python -m glasswing_train.train --max-steps 20000 --out ../checkpoints/main
    uv run python -m glasswing_train.train --resume --out ../checkpoints/main   # after a stop

Every `eval_every` steps: validation loss on a fixed set of windows, a line in metrics.jsonl, a
sample story, and a checkpoint (weights + optimizer state, so a stopped run resumes exactly) plus
model.safetensors for the Rust engine. Ctrl-C saves a checkpoint before exiting.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from glasswing_train.export import save_model
from glasswing_train.model import ModelConfig, Transformer
from glasswing_train.sample import generate
from glasswing_train.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TrainConfig:
    batch_size: int = 64  # sequences per micro-batch
    grad_accum: int = 2  # micro-batches per optimizer step -> 64*2*256 = 32k tokens/step
    seq_len: int = 256
    max_steps: int = 20000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 500
    eval_batches: int = 40
    seed: int = 1337


def lr_at(step: int, c: TrainConfig) -> float:
    """Linear warmup, then cosine decay from lr to min_lr."""
    if step < c.warmup_steps:
        return c.lr * (step + 1) / c.warmup_steps
    progress = min(1.0, (step - c.warmup_steps) / max(1, c.max_steps - c.warmup_steps))
    return c.min_lr + 0.5 * (c.lr - c.min_lr) * (1 + math.cos(math.pi * progress))


class Batches:
    """Random seq_len+1 windows from a token file; x is the window, y the same shifted by one."""

    def __init__(self, path: Path, c: TrainConfig, device: str, seed: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.c, self.device = c, device
        self.rng = np.random.default_rng(seed)

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        T = self.c.seq_len
        starts = self.rng.integers(0, len(self.data) - T - 1, self.c.batch_size)
        w = torch.from_numpy(np.stack([self.data[s : s + T + 1] for s in starts]).astype(np.int64))
        w = w.pin_memory().to(self.device, non_blocking=True)
        return w[:, :-1], w[:, 1:]


def make_optimizer(model: Transformer, c: TrainConfig) -> torch.optim.AdamW:
    # Decay matrices (incl. the embedding table), not the norm weights.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": c.weight_decay},
            {"params": no_decay, "weight_decay": 0},
        ],
        lr=c.lr,
        betas=(0.9, 0.95),
        fused=True,
    )


def loss_fn(model: Transformer, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def evaluate(model: Transformer, val: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    model.eval()
    loss = sum(loss_fn(model, x, y).item() for x, y in val) / len(val)
    model.train()
    return loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train glasswing.")
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints/main")
    parser.add_argument("--resume", action="store_true")
    for field, value in asdict(TrainConfig()).items():
        parser.add_argument(f"--{field.replace('_', '-')}", type=type(value), default=value)
    args = parser.parse_args()
    c = TrainConfig(**{k: getattr(args, k) for k in asdict(TrainConfig())})
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    torch.manual_seed(c.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tok = Tokenizer.load(ROOT / "assets/tokenizer.json")
    model = Transformer(ModelConfig(max_seq_len=c.seq_len))
    model.init_weights(seed=c.seed)
    model.to(device)
    opt = make_optimizer(model, c)
    step = 0

    ckpt_path = out / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        c = TrainConfig(**ckpt["train_config"])
        print(f"resumed at step {step}")
    (out / "config.json").write_text(
        json.dumps({"train": asdict(c), "model": json.loads(model.cfg.to_json())}, indent=2)
    )

    def save(step: int) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step,
                "train_config": asdict(c),
            },
            out / "ckpt.tmp",
        )
        (out / "ckpt.tmp").replace(ckpt_path)  # atomic: a crash mid-write keeps the old one
        save_model(model, out / "model.safetensors")

    train = Batches(ROOT / "data/train.bin", c, device, seed=c.seed + step)
    # Fixed validation windows, so every eval measures the same thing.
    val_src = Batches(ROOT / "data/val.bin", c, device, seed=0)
    val = [val_src.get() for _ in range(c.eval_batches)]

    print(f"{model.num_params():,} params, {c.batch_size * c.grad_accum * c.seq_len:,} tokens/step")
    t0, tokens_since, losses = time.perf_counter(), 0, []
    metrics = (out / "metrics.jsonl").open("a")
    try:
        while step < c.max_steps:
            lr = lr_at(step, c)
            for g in opt.param_groups:
                g["lr"] = lr
            for _ in range(c.grad_accum):
                x, y = train.get()
                loss = loss_fn(model, x, y)
                (loss / c.grad_accum).backward()
                losses.append(loss.item())
                tokens_since += x.numel()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % c.eval_every == 0 or step == c.max_steps:
                elapsed = time.perf_counter() - t0
                row = {
                    "step": step,
                    "train_loss": round(sum(losses) / len(losses), 4),
                    "val_loss": round(evaluate(model, val), 4),
                    "lr": lr,
                    "grad_norm": round(float(grad_norm), 3),
                    "tokens_per_s": round(tokens_since / elapsed),
                    "tokens_seen": step * c.batch_size * c.grad_accum * c.seq_len,
                    "time": time.strftime("%H:%M:%S"),
                }
                model.eval()
                ids = generate(model, tok, "Once upon a time", 60, seed=step)
                model.train()
                row["sample"] = tok.decode(ids)
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                print(json.dumps(row), flush=True)
                save(step)
                t0, tokens_since, losses = time.perf_counter(), 0, []
    except KeyboardInterrupt:
        print(f"interrupted at step {step}, saving")
        save(step)


if __name__ == "__main__":
    main()
