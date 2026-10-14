from pathlib import Path

import pytest

from uvt_core.config import config_path, env_overrides, load_config, parse_env_value, tool_config
from uvt_core.errors import InvalidInputError


def test_parse_env_value_literals():
    assert parse_env_value("true") is True
    assert parse_env_value("3") == 3
    assert parse_env_value('["a", "b"]') == ["a", "b"]
    assert parse_env_value("large-v3-turbo") == "large-v3-turbo"
    assert parse_env_value("C:\\Users\\me") == "C:\\Users\\me"


def test_env_overrides_sections_and_top_level():
    env = {
        "UVT_LOG_LEVEL": "DEBUG",
        "UVT_TRANSCRIBE__MODEL_CPU": "base",
        "UVT_TRANSCRIBE__BEAM_SIZE": "2",
        "UVT_CONFIG_FILE": "/ignored",
        "OTHER": "x",
    }
    assert env_overrides(env) == {
        "log_level": "DEBUG",
        "transcribe": {"model_cpu": "base", "beam_size": 2},
    }


def test_config_file_merged_with_env(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[transcribe]\nmodel_cpu = "small"\nmodel_gpu = "large-v3"\n', encoding="utf-8")
    env = {"UVT_CONFIG_FILE": str(cfg), "UVT_TRANSCRIBE__MODEL_CPU": "tiny"}
    assert config_path(env) == cfg
    assert tool_config("transcribe", env=env) == {"model_cpu": "tiny", "model_gpu": "large-v3"}
    assert tool_config("missing", env=env) == {}


def test_missing_config_file_is_empty(tmp_path: Path):
    assert load_config(env={"UVT_CONFIG_FILE": str(tmp_path / "nope.toml")}) == {}


def test_invalid_config_is_invalid_input(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("not = = toml", encoding="utf-8")
    with pytest.raises(InvalidInputError, match="Invalid config file"):
        load_config(env={"UVT_CONFIG_FILE": str(cfg)})
