from pathlib import Path

from tinystories_scratch.config import TrainConfig, load_config


def test_defaults():
    cfg = TrainConfig()
    assert cfg.device == "cpu"
    assert cfg.n_layer >= 1


def test_load_smoke_yaml():
    path = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
    cfg = load_config(path)
    assert cfg.run_name == "smoke"
    assert cfg.dataset == "synthetic"
    assert cfg.max_iters == 30
