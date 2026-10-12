import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from glasswing_train.export import load_model, save_model
from glasswing_train.fixtures import PROMPTS, TINY, random_tiny_model
from glasswing_train.model import ModelConfig, Transformer, apply_rope, rope_cos_sin

ROOT = Path(__file__).resolve().parents[2]

SMALL = ModelConfig(
    vocab_size=64, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=48, max_seq_len=32
)


def test_causal_no_peeking_at_the_future():
    # Changing a later token must not change any earlier position's logits.
    model = Transformer(SMALL)
    model.init_weights(seed=0)
    a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    b = a.clone()
    b[0, 4] = 9
    with torch.no_grad():
        la, lb = model(a), model(b)
    assert torch.allclose(la[0, :4], lb[0, :4])
    assert not torch.allclose(la[0, 4:], lb[0, 4:])


def test_rope_score_depends_only_on_distance():
    cfg = SMALL
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, cfg.head_dim)
    k = torch.randn(1, 1, 1, cfg.head_dim)
    cos, sin = rope_cos_sin(cfg, torch.arange(32))

    def score(pq, pk):
        rq = apply_rope(q, cos[pq], sin[pq])
        rk = apply_rope(k, cos[pk], sin[pk])
        return (rq * rk).sum()

    assert torch.allclose(score(5, 2), score(25, 22), atol=1e-5)  # same distance, same score
    assert not torch.allclose(score(5, 2), score(5, 4), atol=1e-3)


def test_rope_preserves_length():
    cos, sin = rope_cos_sin(SMALL, torch.arange(10))
    x = torch.randn(1, 2, 10, SMALL.head_dim)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_attention_weights_are_causal_distributions():
    model = random_tiny_model()
    trace: dict = {}
    with torch.no_grad():
        model(torch.tensor([[5, 6, 7, 8]]), trace=trace)
    w = trace["layers.0.attn_weights"][0]  # [H, T, T]
    assert torch.allclose(w.sum(-1), torch.ones(TINY.n_heads, 4))
    assert torch.all(torch.triu(w, diagonal=1) == 0)


def test_can_overfit_one_sentence():
    # If the model can't memorise one short sequence, something in the wiring is broken.
    torch.manual_seed(0)
    model = Transformer(SMALL)
    model.init_weights(seed=0)
    seq = torch.randint(0, SMALL.vocab_size, (1, 17))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(100):
        logits = model(seq[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, SMALL.vocab_size), seq[:, 1:].reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[0] > 3.5  # ~ln(64) = 4.16 at the start: uniform guessing
    assert losses[-1] < 0.1


def test_save_load_roundtrip(tmp_path):
    model = random_tiny_model(seed=1)
    save_model(model, tmp_path / "m.safetensors")
    loaded = load_model(tmp_path / "m.safetensors")
    assert loaded.cfg == model.cfg
    x = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        assert torch.equal(model(x), loaded(x))


def test_parameter_names_follow_llama_layout():
    names = set(random_tiny_model().state_dict())
    assert "model.embed_tokens.weight" in names
    assert "model.layers.1.self_attn.k_proj.weight" in names
    assert "model.layers.0.mlp.gate_proj.weight" in names
    assert "model.norm.weight" in names
    assert not any("lm_head" in n for n in names)  # tied to the embeddings


def test_default_config_size():
    # ~15M parameters as planned in ADR 0001.
    n = Transformer(ModelConfig()).num_params()
    assert 14e6 < n < 16e6, n


@pytest.fixture(scope="module")
def expected():
    with safe_open(str(ROOT / "fixtures/tiny/expected.safetensors"), framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}, json.loads(f.metadata()["prompts"])


def test_golden_fixtures_are_current(expected):
    # The committed fixtures must match what this model code computes from the committed weights.
    # If this fails after a deliberate model change, regenerate with
    # `uv run python -m glasswing_train.fixtures` and say why in the commit.
    tensors, prompts = expected
    assert prompts == PROMPTS
    model = load_model(ROOT / "fixtures/tiny/model.safetensors")
    for i in range(len(prompts)):
        trace: dict = {}
        with torch.no_grad():
            model(tensors[f"p{i}.tokens"][None], trace=trace)
        for name, t in trace.items():
            torch.testing.assert_close(
                t[0],
                tensors[f"p{i}.{name}"],
                rtol=1e-5,
                atol=1e-5,
                msg=lambda m, n=f"p{i}.{name}": f"{n}: {m}",
            )
