from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

from category_priors.object_scope.alpha_clip_model import MODEL_VERSION, validate_model_spec, load_local_alpha_clip, _adapt_clip_base
from category_priors.object_scope.artifacts import file_digest


def spec_fixture(tmp_path):
    package = tmp_path / "alpha_clip"
    package.mkdir()
    for name in ("__init__.py", "alpha_clip.py", "model.py", "simple_tokenizer.py"):
        (package / name).write_text("# CPU synthetic asset only\n")
    tokenizer = package / "bpe_simple_vocab_16e6.txt.gz"
    tokenizer.write_bytes(b"synthetic")
    base, alpha = tmp_path / "base.pt", tmp_path / "alpha.pth"
    base.write_bytes(b"synthetic-base")
    alpha.write_bytes(b"synthetic-alpha")
    ref = lambda path: {"path": str(path), "sha256": file_digest(path)}
    return dict(schema=MODEL_VERSION, variant="ViT-L/14@336px", training="GRIT20M-full-tune-4xe",
        base_checkpoint=ref(base), alpha_checkpoint=ref(alpha), package_root=str(package),
        source_files=[ref(p) for p in sorted(package.glob("*.py"))], tokenizer_file=ref(tokenizer), precision="float16")


def test_local_complete_assets_validate_without_torch_or_loading(tmp_path):
    spec = spec_fixture(tmp_path)
    before = "torch" in sys.modules
    result = validate_model_spec(spec)
    assert result["precision"] == "float16"
    assert ("torch" in sys.modules) == before


@pytest.mark.parametrize("change", ["extra-source", "omitted-source", "tampered-weight", "variant", "extra-field", "tokenizer"])
def test_asset_and_variant_rejections(tmp_path, change):
    spec = spec_fixture(tmp_path)
    if change == "extra-source":
        (Path(spec["package_root"]) / "unregistered.py").write_text("# changed\n")
    elif change == "omitted-source":
        spec["source_files"].pop()
    elif change == "tampered-weight":
        Path(spec["alpha_checkpoint"]["path"]).write_bytes(b"different")
    elif change == "variant":
        spec["variant"] = "ViT-B/16"
    elif change == "extra-field":
        spec["allow_download"] = True
    else:
        spec["tokenizer_file"] = spec["base_checkpoint"]
    with pytest.raises(ValueError):
        validate_model_spec(spec)


def test_model_factory_cannot_load_without_reviewed_lease(tmp_path, monkeypatch):
    spec = spec_fixture(tmp_path)
    monkeypatch.delenv("SAGA_OBJECT_LEASE_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="execution lease"):
        load_local_alpha_clip(spec, device="cpu")


def test_explicit_attention_conversion_not_generic_strict_false():
    class FakeTorch:
        zeros_like = staticmethod(np.zeros_like)
    state = {"visual.conv1.weight": np.ones((4, 3, 2, 2)),
             "visual.transformer.attn.in_proj_weight": np.ones((4, 4)),
             "transformer.attn.in_proj_weight": np.ones((4, 4)), "input_resolution": 336}
    result = _adapt_clip_base(state, FakeTorch)
    assert "visual.transformer.attn.in_proj.weight" in result
    assert "transformer.attn.in_proj_weight" in result
    assert result["visual.conv1_alpha.weight"].shape == (4, 1, 2, 2)
    assert "input_resolution" not in result
    with pytest.raises(ValueError, match="already has"):
        _adapt_clip_base(result, FakeTorch)


def test_alpha_reload_clock_is_not_part_of_model_or_region_cache_identity(tmp_path, monkeypatch):
    from category_priors.object_scope import alpha_clip_model as ac
    from category_priors.object_scope.artifacts import digest
    config=spec_fixture(tmp_path)
    class Model:
        def __init__(self, **kwargs):self.visual=self
        def load_state_dict(self, state, strict):assert strict is True
        def eval(self):return self
        def requires_grad_(self, value):return self
        def to(self, device):return self
    alias="_synthetic_alpha_reload"
    module=SimpleNamespace(__file__=str(Path(config["package_root"])/"model.py"),CLIP=Model,convert_weights=lambda model:None)
    package=SimpleNamespace(__file__=str(Path(config["package_root"])/"__init__.py"),tokenize=lambda *a:None)
    monkeypatch.setitem(sys.modules,alias,package)
    monkeypatch.setitem(sys.modules,alias+".model",module)
    monkeypatch.setattr(ac,"_import_package",lambda *args:(package,alias))
    base={"visual.conv1.weight":np.zeros((4,3,2,2))}
    torch=SimpleNamespace(__version__="CPU-synthetic",_C=SimpleNamespace(__file__=__file__),
        version=SimpleNamespace(cuda=None),zeros_like=np.zeros_like,
        jit=SimpleNamespace(load=lambda *a,**kw:SimpleNamespace(state_dict=lambda:base)),
        load=lambda *a,**kw:{"alpha":1},
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(allow_tf32=False)),are_deterministic_algorithms_enabled=lambda:True)
    monkeypatch.setitem(sys.modules,"torch",torch)
    monkeypatch.setenv("SAGA_OBJECT_LEASE_TOKEN","CPU-synthetic-only")
    clock=iter((10.,12.,100.,119.))
    monkeypatch.setattr(ac.time,"perf_counter",lambda:next(clock))
    first=ac.load_local_alpha_clip(config,device="cpu")
    second=ac.load_local_alpha_clip(config,device="cpu")
    assert first.load_seconds==2. and second.load_seconds==19.
    assert digest(first.runtime_identity)==digest(second.runtime_identity)
    assert "load_seconds" not in first.runtime_identity
