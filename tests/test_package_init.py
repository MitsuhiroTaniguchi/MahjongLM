from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from unittest.mock import patch
import sys
import pytest


def test_all_omits_huggingface_exports_without_optional_dependencies() -> None:
    module = _load_tenhou_tokenizer_without_huggingface_deps()

    assert "Vocabulary" in module.__all__
    assert "save_year_hf_dataset" not in module.__all__
    assert "MahjongTokenizerFast" not in module.__all__


def test_optional_huggingface_exports_raise_attribute_error_without_dependencies() -> None:
    module = _load_tenhou_tokenizer_without_huggingface_deps()

    assert hasattr(module, "save_year_hf_dataset") is False
    with pytest.raises(AttributeError):
        getattr(module, "save_year_hf_dataset")


def _load_tenhou_tokenizer_without_huggingface_deps():
    init_path = Path(__file__).resolve().parents[1] / "src" / "tenhou_tokenizer" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "tenhou_tokenizer_no_hf",
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None):
        if name in {"datasets", "transformers"}:
            return None
        return real_find_spec(name, package)

    with patch("importlib.util.find_spec", side_effect=fake_find_spec):
        assert spec.loader is not None
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)

    return module
