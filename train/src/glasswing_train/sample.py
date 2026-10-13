"""Text generation with the PyTorch model: used for samples during training and as the
reference the Rust sampler is compared against."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from glasswing_train.model import Transformer
from glasswing_train.tokenizer import Tokenizer


@torch.no_grad()
def generate(
    model: Transformer,
    tok: Tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    seed: int = 0,
) -> list[int]:
    """Returns the generated token ids (prompt excluded). temperature 0 = greedy.
    Stops at <|endoftext|> or the context limit."""
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    ids = [tok.special_id, *tok.encode(prompt)]
    out: list[int] = []
    for _ in range(max_new_tokens):
        if len(ids) >= model.cfg.max_seq_len:
            break
        logits = model(torch.tensor([ids], device=device))[0, -1].float()
        if temperature == 0:
            nxt = int(logits.argmax())
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            sorted_p, order = probs.sort(descending=True)
            # Keep the smallest set of tokens whose probabilities add up to top_p.
            keep = sorted_p.cumsum(0) - sorted_p < top_p
            sorted_p = sorted_p * keep
            nxt = int(order[torch.multinomial(sorted_p / sorted_p.sum(), 1, generator=gen)])
        if nxt == tok.special_id:
            break
        ids.append(nxt)
        out.append(nxt)
    return out
