import torch

from tinystories_scratch.model import GPT


def test_forward_shapes_and_loss():
    model = GPT(vocab_size=32, n_layer=2, n_head=2, n_embd=32, block_size=16, dropout=0.0)
    x = torch.randint(0, 32, (2, 16))
    y = torch.randint(0, 32, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 32)
    assert loss is not None
    assert torch.isfinite(loss)


def test_generate_grows_sequence():
    model = GPT(vocab_size=20, n_layer=1, n_head=2, n_embd=16, block_size=8, dropout=0.0)
    model.eval()
    idx = torch.zeros(1, 2, dtype=torch.long)
    out = model.generate(idx, max_new_tokens=5, temperature=1.0, top_k=5)
    assert out.shape == (1, 7)


def test_param_count_positive():
    model = GPT(vocab_size=50, n_layer=2, n_head=2, n_embd=32, block_size=32)
    assert model.count_parameters() > 1000
