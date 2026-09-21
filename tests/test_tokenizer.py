from tinystories_scratch.tokenizer import CharTokenizer


def test_roundtrip():
    tok = CharTokenizer.from_texts(["hello", "world"])
    text = "hello"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_unk_and_pad():
    tok = CharTokenizer.from_texts(["ab"])
    assert tok.pad_id == 0
    assert tok.unk_id == 1
    assert tok.unk_id in tok.encode("z")


def test_serialize():
    tok = CharTokenizer.from_texts(["cat"])
    restored = CharTokenizer.from_dict(tok.to_dict())
    assert restored.vocab_size == tok.vocab_size
    assert restored.encode("ca") == tok.encode("ca")
