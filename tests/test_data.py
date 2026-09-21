from tinystories_scratch.data import (
    SYNTHETIC_STORIES,
    build_tokenizer_and_datasets,
    expand_synthetic,
)


def test_expand_synthetic_count():
    texts = expand_synthetic(25, seed=1)
    assert len(texts) == 25
    assert all(isinstance(t, str) and t for t in texts)


def test_build_datasets():
    texts = SYNTHETIC_STORIES[:8]
    tok, train_ds, val_ds = build_tokenizer_and_datasets(
        texts, block_size=32, val_ratio=0.25, seed=0
    )
    assert tok.vocab_size > 5
    assert len(train_ds) >= 1
    assert len(val_ds) >= 1
    x, y = train_ds[0]
    assert x.shape == (32,)
    assert y.shape == (32,)
    assert (x[1:] == y[:-1]).all()
